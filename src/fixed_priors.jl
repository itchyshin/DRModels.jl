using SpecialFunctions: loggamma

"Supertype of supported distributional-model prior specifications."
abstract type AbstractDrmPrior end

"A normal or ridge penalty on coefficients of one distributional predictor."
struct FixedPrior <: AbstractDrmPrior
    kind::Symbol
    dpar::Symbol
    coef::String
    intercept::Union{Nothing,Symbol}
    mean::Float64
    precision::Float64
    constant::Float64
end

"A Gamma density with respect to a named random-intercept SD."
struct SDGammaPrior <: AbstractDrmPrior
    dpar::Symbol
    group::String
    coef::String
    shape::Float64
    rate::Float64
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

_sd_prior_value(row, theta) = row.rate * exp(theta[row.index]) - (row.shape - 1) * theta[row.index] - row.shape * log(row.rate) + loggamma(row.shape)

function _fixed_prior(kind, dpar, coef, intercept, mean, precision, constant)
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
    return FixedPrior(kind, d, c, inc, Float64(mean), Float64(precision), Float64(constant))
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
            push!(rows, (dpar=prior.dpar, coef=nms[i], index=offsets[prior.dpar]+i,
                         matrix_column=i, mean=prior.mean, sd=prior.precision == 0 ? Inf : inv(sqrt(prior.precision)), precision=prior.precision, constant=prior.constant,
                         kind=prior.kind, reason=prior.coef == "all" ? :all : :exact))
        end
    end
    rows
end

function _fixed_prior_value(rows, theta)
    sum((0.5 * r.precision * (theta[r.index] - r.mean)^2 + r.constant for r in rows); init=zero(eltype(theta)))
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

"""
    resolve_priors(formula, family; data, priors)

Preview the exact fitted columns and prior precision without optimizing. The
route covers complete-response Gaussian models with fixed effects and at most one
mean random intercept, and fixed-effect binomial logit models. Priors refer to
predictor coefficients and depend on units and coding.
"""
function resolve_priors(f, fam; data, priors)
    f isa DrmFormula && (fam isa Gaussian || (fam isa Binomial && fam.link === :logit)) ||
        throw(ArgumentError("resolve_priors supports Gaussian and fixed-effect binomial logit models"))
    _has_joint_mi(f) && throw(ArgumentError("resolve_priors does not support missing-predictor models"))
    isempty(_sd_parts(f)) && isempty(_sdphylo_parts(f)) || throw(ArgumentError("resolve_priors does not support SD formulas"))
    for response in _formula_responses(f)
        response === nothing && continue
        any(p -> p isa SDGammaPrior, priors) && any(ismissing, _table_column(data, response)) && throw(ArgumentError("SD priors require complete responses"))
        all(_observed_response_mask(_table_column(data, response))) || throw(ArgumentError("resolve_priors requires complete responses"))
    end
    _reject_sd_missing_predictors(f, data, priors)
    dat = fam isa Gaussian ? _drop_missing_predictor_rows(f, data) : data
    length(_table_column(dat, f.response)) == length(_table_column(data, f.response)) ||
        throw(ArgumentError("resolve_priors requires complete predictors"))
    forms = Dict(f.forms); dpars = fam isa Gaussian ? (:mu, :sigma) : (:mu,)
    names = Pair{Symbol,Vector{String}}[]; intercepts = Dict{Symbol,Union{Nothing,Int}}()
    cache = Ref(Dict{Symbol,Any}())
    sd_group = nothing
    for d in dpars
        rhs, re, meta, structured, structured_slope, temporal_term, structured_sd_slope = _split_ranef(forms[d]; allow_temporal=true)
        allowed_re = fam isa Gaussian && d === :mu && length(re) == 1 && _re_kind(first(re)[1])[1] === :intercept
        (isempty(re) || allowed_re) && meta === nothing && structured === nothing && structured_slope === nothing && temporal_term === nothing && structured_sd_slope === nothing || throw(ArgumentError("resolve_priors supports only one Gaussian mean random intercept"))
        if allowed_re
            grp = first(re)[2]
            sd_group = grp
            _group_index(_table_column(dat, grp); col=grp)
        end
        rhs, _ = _extract_offset(rhs)
        _, X, nms = _design(f.response, rhs, dat; schema_cache=cache, schema_key=d)
        rank(X) == size(X, 2) || throw(ArgumentError("resolve_priors requires full-rank design matrices"))
        push!(names, d => nms)
        intercepts[d] = _fixed_intercept_index(f, d, rhs, nms; cache=cache)
    end
    priors isa AbstractVector && all(p -> p isa Union{FixedPrior,SDGammaPrior}, priors) || throw(ArgumentError("priors must be created by drm_priors"))
    fixed = FixedPrior[p for p in priors if p isa FixedPrior]
    rows = _resolve_fixed_priors(fixed, names, intercepts)
    sd = _resolve_sd_prior(priors, sd_group, sum(length(last(n)) for n in names) + 1)
    sd === nothing ? rows : vcat(rows, [sd])
end
