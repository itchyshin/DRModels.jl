using SpecialFunctions: loggamma

"Supertype of supported distributional-model prior specifications."
abstract type AbstractDrmPrior end

"A normal, Student-t, or ridge penalty on coefficients of one distributional predictor."
struct FixedPrior <: AbstractDrmPrior
    kind::Symbol
    dpar::Symbol
    coef::String
    intercept::Union{Nothing,Symbol}
    mean::Float64
    precision::Float64
    constant::Float64
    df::Union{Nothing,Float64}
end

# Preserve the internal constructor used by existing normal/ridge and SD-prior
# paths; `df` is meaningful only for `kind === :student_t`.
FixedPrior(kind::Symbol, dpar::Symbol, coef::String, intercept::Union{Nothing,Symbol},
           mean::Float64, precision::Float64, constant::Float64) =
    FixedPrior(kind, dpar, coef, intercept, mean, precision, constant, nothing)

"A Gamma density with respect to a named random-intercept SD."
struct SDGammaPrior <: AbstractDrmPrior
    dpar::Symbol
    group::String
    coef::String
    shape::Float64
    rate::Float64
end

"A Gamma density with respect to the natural residual SD of a Gaussian model."
struct ResidualSDGammaPrior <: AbstractDrmPrior
    mean::Float64
    shape::Float64
    rate::Float64
end

function _residual_sd_gamma_prior(mean, shape)
    mean isa Real && isfinite(mean) && mean > 0 ||
        throw(ArgumentError("residual SD Gamma mean must be finite and positive"))
    shape isa Real && isfinite(shape) && shape > 1 ||
        throw(ArgumentError("residual SD Gamma shape must be finite and greater than 1"))
    μ, a = Float64(mean), Float64(shape)
    rate = a / μ
    isfinite(rate) && rate > 0 ||
        throw(ArgumentError("residual SD Gamma rate must be finite and positive"))
    ResidualSDGammaPrior(μ, a, rate)
end

"""`drm_prior_sd_gamma(; dpar, group, coef="(Intercept)", shape, rate)` specifies a Gamma density on the natural random-intercept SD (shape > 1, rate > 0)."""
function drm_prior_sd_gamma(; dpar, group, coef="(Intercept)", shape, rate)
    d = Symbol(dpar); g = String(group); c = String(coef)
    d === :mu || throw(ArgumentError("SD Gamma prior requires dpar=mu"))
    isempty(g) && throw(ArgumentError("SD Gamma prior requires an exact group name"))
    c == "(Intercept)" || throw(ArgumentError("SD Gamma prior supports only coefficient (Intercept)"))
    shape isa Real && isfinite(shape) && shape > 1 || throw(ArgumentError("SD Gamma prior shape must be finite and greater than 1"))
    rate isa Real && isfinite(rate) && rate > 0 || throw(ArgumentError("SD Gamma prior rate must be finite and positive"))
    a = Float64(shape); b = Float64(rate)
    isfinite(a) && isfinite(b) && a > 1 && b > 0 || throw(ArgumentError("SD Gamma prior parameters are outside the representable range"))
    return SDGammaPrior(d, g, c, a, b)
end

function _resolve_sd_prior(priors, grp, index)
    sds = filter(p -> p isa SDGammaPrior, priors)
    isempty(sds) && return nothing
    length(sds) == 1 || throw(ArgumentError("duplicate or ambiguous SD Gamma priors"))
    p = only(sds)
    p.dpar === :mu || throw(ArgumentError("SD Gamma prior requires dpar=mu"))
    p.coef == "(Intercept)" || throw(ArgumentError("SD Gamma prior supports only coefficient (Intercept)"))
    isempty(p.group) && throw(ArgumentError("SD Gamma prior requires an exact group name"))
    isfinite(p.shape) && p.shape > 1 || throw(ArgumentError("SD Gamma prior shape must be finite and greater than 1"))
    isfinite(p.rate) && p.rate > 0 || throw(ArgumentError("SD Gamma prior rate must be finite and positive"))
    grp !== nothing && p.group == String(grp) || throw(ArgumentError("SD Gamma prior group $(p.group) does not match the unique Gaussian mean random intercept"))
    return (class=:random_effect_sd, dpar=:mu, group=p.group, block=:iid_intercept,
            coef=p.coef, scale=:sd, shape=p.shape, rate=p.rate, index=index)
end

# Resolve by group identity, not prior-list order. Reuse the single-target
# validator so constructor bypasses receive the same checks as the q1 route.
function _resolve_sd_priors(priors, groups, first_index)
    length(unique(groups)) == length(groups) || throw(ArgumentError("ambiguous repeated random-intercept group"))
    rows = NamedTuple[]
    seen = Set{String}()
    for p in priors
        p isa SDGammaPrior || continue
        p.group in seen && throw(ArgumentError("duplicate SD Gamma prior for $(p.group)"))
        push!(seen, p.group)
        i = findfirst(g -> String(g) == p.group, groups)
        i === nothing && throw(ArgumentError("SD Gamma prior group $(p.group) does not match a Gaussian mean random intercept"))
        push!(rows, _resolve_sd_prior([p], groups[i], first_index + i - 1))
    end
    rows
end

_sd_prior_value(row, theta) = row.rate * exp(theta[row.index]) - (row.shape - 1) * theta[row.index] - row.shape * log(row.rate) + loggamma(row.shape)

function _fixed_prior(kind, dpar, coef, intercept, mean, precision, constant; df=nothing)
    d = Symbol(dpar); c = String(coef)
    isempty(c) && throw(ArgumentError("prior coefficient must not be empty"))
    inc = intercept === nothing ? nothing : Symbol(intercept)
    if c == "all"
        inc in (:include, :exclude) || throw(ArgumentError("coef = all requires intercept = include or exclude"))
    elseif inc !== nothing
        throw(ArgumentError("intercept is only valid with coef = all"))
    end
    isfinite(mean) && isfinite(precision) && precision >= 0 ||
        throw(ArgumentError("prior mean and precision must be finite; precision must be nonnegative"))
    if kind === :student_t
        df isa Real && isfinite(df) && df > 0 && precision > 0 && isfinite(constant) ||
            throw(ArgumentError("Student-t prior requires finite positive scale and degrees of freedom"))
    else
        df === nothing || throw(ArgumentError("degrees of freedom are only valid for a Student-t prior"))
    end
    return FixedPrior(kind, d, c, inc, Float64(mean), Float64(precision), Float64(constant),
                      df === nothing ? nothing : Float64(df))
end

"""`drm_prior_normal(; dpar, coef, mean=0, sd, intercept=nothing)` specifies a normal coefficient density."""
function drm_prior_normal(; dpar, coef, mean=0, sd, intercept=nothing)
    sd isa Real && isfinite(sd) && sd > 0 || throw(ArgumentError("normal prior sd must be finite and positive"))
    precision = inv(Float64(sd)^2)
    isfinite(precision) && precision > 0 || throw(ArgumentError("normal prior sd is outside the representable precision range"))
    _fixed_prior(:normal, dpar, coef, intercept, mean, precision, log(sd) + 0.5log(2π))
end

"""`drm_prior_ridge(; dpar, coef, lambda, intercept=nothing)` specifies `lambda * beta^2 / 2`."""
function drm_prior_ridge(; dpar, coef, lambda, intercept=nothing)
    _fixed_prior(:ridge, dpar, coef, intercept, 0, lambda, 0)
end

"""`drm_prior_student_t(; dpar, coef, location=0, scale, df, intercept=nothing)` specifies a normalized Student-t coefficient density."""
function drm_prior_student_t(; dpar, coef, location=0, scale, df, intercept=nothing)
    scale isa Real && isfinite(scale) && scale > 0 ||
        throw(ArgumentError("Student-t prior scale must be finite and positive"))
    df isa Real && isfinite(df) && df > 0 ||
        throw(ArgumentError("Student-t prior degrees of freedom must be finite and positive"))
    s = Float64(scale); ν = Float64(df)
    precision = inv(s^2)
    isfinite(precision) && precision > 0 ||
        throw(ArgumentError("Student-t prior scale is outside the representable precision range"))
    isfinite(location) || throw(ArgumentError("Student-t prior location must be finite"))
    normalizer = log(s) + 0.5 * log(ν * π) + loggamma(ν / 2) - loggamma((ν + 1) / 2)
    isfinite(normalizer) || throw(ArgumentError("Student-t prior normalizer is not finite"))
    _fixed_prior(:student_t, dpar, coef, intercept, location, precision, normalizer; df=ν)
end

"""`drm_priors(p...)` collects fixed-effect and supported random-effect SD priors for `drm(...; priors=...)`."""
drm_priors(p::FixedPrior...) = collect(p)
drm_priors(p::AbstractDrmPrior...) = collect(p)

function _resolve_fixed_priors(priors, names, intercepts)
    priors isa AbstractVector{<:FixedPrior} || throw(ArgumentError("priors must be created by drm_priors"))
    blocks = Dict(names); offsets = Dict{Symbol,Int}(); offset = 0
    for (d, n) in names
        offsets[d] = offset; offset += length(n)
    end
    rows = NamedTuple[]; seen = Set{Tuple{Symbol,Int}}()
    for prior in priors
        haskey(blocks, prior.dpar) || throw(ArgumentError("unknown prior dpar $(prior.dpar)"))
        nms = blocks[prior.dpar]; iint = get(intercepts, prior.dpar, nothing)
        idx = prior.coef == "all" ? [i for i in eachindex(nms) if prior.intercept === :include || i != iint] : findall(==(prior.coef), nms)
        isempty(idx) && throw(ArgumentError("prior selection $(prior.dpar):$(prior.coef) matched no coefficient"))
        for i in idx
            key = (prior.dpar, i)
            key in seen && throw(ArgumentError("overlapping prior selection for $(prior.dpar):$(nms[i])"))
            push!(seen, key)
            push!(rows, (class=:fixef, scale=:link, dpar=prior.dpar, coef=nms[i], index=offsets[prior.dpar]+i,
                         matrix_column=i, mean=prior.mean, sd=prior.precision == 0 ? Inf : inv(sqrt(prior.precision)), precision=prior.precision, constant=prior.constant,
                         kind=prior.kind, location=prior.mean,
                         scale_parameter=prior.kind === :student_t ? inv(sqrt(prior.precision)) : nothing,
                         df=prior.df, reason=prior.coef == "all" ? :all : :exact))
        end
    end
    rows
end

function _fixed_prior_value(rows, theta)
    sum((r.kind === :residual_sd_gamma ?
         r.rate * exp(theta[r.index]) - (r.shape - 1) * theta[r.index] -
             r.shape * log(r.rate) + loggamma(r.shape) :
         r.kind === :student_t ?
         r.constant + 0.5 * (r.df + 1) * log1p(r.precision * (theta[r.index] - r.mean)^2 / r.df) :
         0.5 * r.precision * (theta[r.index] - r.mean)^2 + r.constant for r in rows);
        init=zero(eltype(theta)))
end

function _resolve_residual_sd_prior(priors, names)
    targets = filter(p -> p isa ResidualSDGammaPrior, priors)
    isempty(targets) && return nothing
    length(targets) == 1 || throw(ArgumentError("duplicate or ambiguous residual SD Gamma priors"))
    blocks = Dict(names)
    haskey(blocks, :sigma) && blocks[:sigma] == ["(Intercept)"] ||
        throw(ArgumentError("residual SD Gamma priors require an intercept-only Gaussian sigma model"))
    offsets = Dict{Symbol,Int}(); offset = 0
    for (d, nms) in names
        offsets[d] = offset
        offset += length(nms)
    end
    p = only(targets)
    row = (class=:residual_sd, kind=:residual_sd_gamma, dpar=:sigma,
           coef="(Intercept)", scale=:natural_sd, mean=p.mean,
           shape=p.shape, rate=p.rate, index=offsets[:sigma] + 1)
    return row
end

# Inspect the applied StatsModels terms, rather than guessing intercepts from
# factor columns or substrings in their labels.
function _fixed_intercept_index(f, dpar, rhs, names; cache=f.schema_cache)
    applied = apply_schema(FormulaTerm(Term(f.response), rhs), cache[][dpar], StatisticalModel)
    StatsModels.hasintercept(applied) || return nothing
    i = findfirst(==("(Intercept)"), names)
    i === nothing && throw(ArgumentError("cannot identify the fitted intercept for $dpar"))
    return i
end

function _reject_sd_missing_predictors(f, data, priors)
    any(p -> p isa SDGammaPrior, priors) || return nothing
    Tables.istable(data) || return nothing
    cols = Tables.columntable(data)
    for c in _predictor_columns(f)
        haskey(cols, c) && any(ismissing, cols[c]) &&
            throw(ArgumentError("SD priors require complete predictors (missing in $c)"))
    end
    return nothing
end

include("prior_table.jl")

"""
    resolve_priors(formula, family; data, priors)

Preview the exact fitted columns and prior precision without optimizing. The
route covers complete-response Gaussian models with fixed effects and up to two
distinct mean random intercepts, and fixed-effect binomial logit models. Normal
coefficient priors also support a Gaussian phylogenetic mean intercept with a
matching phylogenetic SD regression. Preview checks group-level SD predictors
but tree-label matching requires fitting with a tree. Priors refer to predictor
coefficients and depend on units and coding. `priors` may also be a
Tables.jl-compatible prior table with `prior`, `class`, and `coef` columns;
see the prior guide for supported distributions, selectors, and scale conventions.
"""
function resolve_priors(f, fam; data, priors)
    table_rows = Tables.istable(priors) ? _parse_prior_table(priors) : nothing
    sd_intent = table_rows !== nothing && any(r -> r.class == "ranef_sd", table_rows)
    residual_sd_intent = table_rows !== nothing && any(r -> r.class == "residual_sd", table_rows)
    f isa DrmFormula && (fam isa Gaussian || (fam isa Binomial && fam.link === :logit)) ||
        throw(ArgumentError("resolve_priors supports Gaussian and fixed-effect binomial logit models"))
    residual_sd_intent && !(fam isa Gaussian) &&
        throw(ArgumentError("residual_sd priors require a Gaussian model"))
    _has_joint_mi(f) && throw(ArgumentError("resolve_priors does not support missing-predictor models"))
    normal_only = table_rows === nothing ?
        (priors isa AbstractVector && all(p -> p isa FixedPrior && p.kind === :normal, priors)) :
        all(r -> r.class == "fixef" && r.kind === :normal, table_rows)
    forms = Dict(f.forms)
    markers = haskey(forms, :mu) ? _collect_structured(forms[:mu]) : Tuple{Symbol,Symbol}[]
    phylo_preview = fam isa Gaussian && normal_only && length(f.forms) == 3 &&
        isempty(_sd_parts(f)) && length(_sdphylo_parts(f)) == 1 && length(markers) == 1 &&
        only(markers)[1] === :phylo && first(only(_sdphylo_parts(f))) === only(markers)[2]
    (isempty(_sd_parts(f)) && isempty(_sdphylo_parts(f))) || phylo_preview ||
        throw(ArgumentError("resolve_priors does not support SD formulas"))
    for response in _formula_responses(f)
        response === nothing && continue
        (sd_intent || (table_rows === nothing && any(p -> p isa SDGammaPrior, priors))) && any(ismissing, _table_column(data, response)) && throw(ArgumentError("SD priors require complete responses"))
        _complete_response(_table_column(data, response)) || throw(ArgumentError("resolve_priors requires complete responses"))
    end
    if sd_intent
        cols = Tables.columntable(data)
        for c in _predictor_columns(f)
            haskey(cols, c) && any(ismissing, cols[c]) && throw(ArgumentError("SD priors require complete predictors (missing in $c)"))
        end
    elseif table_rows === nothing
        _reject_sd_missing_predictors(f, data, priors)
    end
    dat = fam isa Gaussian ? _drop_missing_predictor_rows(f, data) : data
    length(_table_column(dat, f.response)) == length(_table_column(data, f.response)) ||
        throw(ArgumentError("resolve_priors requires complete predictors"))
    if phylo_preview
        grp, sd_rhs = only(_sdphylo_parts(f))
        idx, n_groups = _group_index(_table_column(dat, grp); col=grp)
        _sd_group_design(f.response, sd_rhs, dat, idx, n_groups, grp)
    end
    forms = Dict(f.forms); dpars = fam isa Gaussian ? (:mu, :sigma) : (:mu,)
    names = Pair{Symbol,Vector{String}}[]; intercepts = Dict{Symbol,Union{Nothing,Int}}()
    cache = Ref(Dict{Symbol,Any}())
    sd_groups = Symbol[]
    sd_group_indices = Vector{Int}[]
    for d in dpars
        rhs, re, meta, structured, structured_slope, temporal_term, structured_sd_slope = _split_ranef(forms[d]; allow_temporal=true)
        allowed_re = fam isa Gaussian && d === :mu && 1 <= length(re) <= 2 && all(r -> _re_kind(r[1])[1] === :intercept, re)
        allowed_phylo = phylo_preview && d === :mu && structured == only(markers) && isempty(re)
        (isempty(re) || allowed_re) && meta === nothing &&
            (structured === nothing || allowed_phylo) && structured_slope === nothing &&
            temporal_term === nothing && structured_sd_slope === nothing ||
            throw(ArgumentError("resolve_priors supports up to two distinct Gaussian mean random intercepts or one phylogenetic SD-regression route"))
        if allowed_re
            for (_, grp) in re
                push!(sd_groups, grp)
                idx, _ = _group_index(_table_column(dat, grp); col=grp)
                push!(sd_group_indices, idx)
            end
            length(sd_group_indices) == 2 && sd_group_indices[1] == sd_group_indices[2] && throw(ArgumentError("two random-intercept groups have identical partitions and their SDs cannot be identified separately"))
        end
        rhs, _ = _extract_offset(rhs)
        _, X, nms = _design(f.response, rhs, dat; schema_cache=cache, schema_key=d)
        rank(X) == size(X, 2) || throw(ArgumentError("resolve_priors requires full-rank design matrices"))
        push!(names, d => nms)
        intercepts[d] = _fixed_intercept_index(f, d, rhs, nms; cache=cache)
    end
    table_rows === nothing || (priors = _prior_table_specs(table_rows, String.(sd_groups)))
    priors isa AbstractVector && all(p -> p isa Union{FixedPrior,SDGammaPrior,ResidualSDGammaPrior}, priors) || throw(ArgumentError("priors must be created by drm_priors"))
    fixed = FixedPrior[p for p in priors if p isa FixedPrior]
    rows = _resolve_fixed_priors(fixed, names, intercepts)
    table_rows !== nothing && any(r -> r.coef == "all", rows) &&
        throw(ArgumentError("prior tables cannot select a coefficient literally named all"))
    sds = _resolve_sd_priors(priors, sd_groups, sum(length(last(n)) for n in names) + 1)
    residual = _resolve_residual_sd_prior(priors, names)
    residual === nothing || !any(r -> r.index == residual.index, rows) ||
        throw(ArgumentError("residual_sd prior overlaps a fixed-effect prior on sigma:(Intercept)"))
    vcat(rows, sds, residual === nothing ? NamedTuple[] : [residual])
end
