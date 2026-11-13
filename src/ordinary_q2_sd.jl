# Ordinary (non-structured) correlated intercept-plus-slope with one SD formula
# per coefficient, for Poisson and NegBinomial2:
#
#   y ~ x + (1 + x | p | g)   (labelled)   or   y ~ x + (1 + x | g)   (Poisson)
#   sd(g, coef(intercept)) ~ z0,   sd(g, coef(x)) ~ z1
#
#   eta_i = X_i b + s0_g u0_g + x_i s1_g u1_g,  (u0_g, u1_g) ~ N(0, R(rho)) iid over g,
#   s_k = exp(Z_k alpha_k),  rho = 0.999999 tanh(eta_rho)  (drmTMB's bound).
#
# This is the animal/relmat q2 model with the identity kernel (Q = I, logdet Q = 0),
# so the existing Laplace fit helpers are reused unchanged; the free correlation
# is the block `:recov`, coefficient name "g:atanh_rho". The label `p` is a pure
# label, never a column. NB2 unlabelled `(1 + x | g)` stays on the two-field
# route (twofield_sd_laplace.jl); Gamma ordinary correlated slopes stay refused
# (native drmTMB refuses them too). Native drmTMB 0.8.1 refuses the LABELLED
# ordinary form for Poisson and NB2 outright and fits only the unlabelled
# `(1 + x | g)`, which is the same model.

# Find an ordinary `(1 + x | g)` or labelled `(1 + x | p | g)` bar term in a mean rhs.
# Returns `(; term, group, slope, label)` (`label === nothing` when unlabelled) or
# `nothing`. StatsModels parses `(1 + x | p | g)` left-associatively, `((1 + x) | p) | g`.
function _ordinary_q2_block(rhs)
    rhs === nothing && return nothing
    terms = rhs isa Tuple ? collect(rhs) : Any[rhs]
    for t in terms
        (t isa FunctionTerm && t.f === (|) && length(t.args) == 2 && t.args[2] isa Term) || continue
        inner, label = t.args[1], nothing
        if inner isa FunctionTerm && inner.f === (|)
            (length(inner.args) == 2 && inner.args[2] isa Term) || continue
            inner, label = inner.args[1], inner.args[2].sym
        end
        lhs = _implicit_re_intercept(inner)
        (lhs isa FunctionTerm && lhs.f === (+) && length(lhs.args) == 2) || continue
        consts = filter(x -> x isa ConstantTerm, lhs.args)
        vars = filter(x -> x isa Term, lhs.args)
        (length(consts) == 1 && consts[1].n == 1 && length(vars) == 1) || continue
        return (; term = t, group = t.args[2].sym, slope = vars[1].sym, label)
    end
    return nothing
end

# Route test for the count SD fitters: an ordinary q2 block that carries iid
# coefficient SD targets. Poisson takes the unlabelled and labelled forms; NB2
# takes the labelled form, and the unlabelled form when only ONE coefficient
# carries an SD formula (two SD formulas on the unlabelled form stay on the
# two-field route).
function _ordinary_q2_sd_block(f::DrmFormula, fam::Union{Poisson,NegBinomial2})
    block = _ordinary_q2_block(get(Dict(f.forms), :mu, nothing))
    block === nothing && return nothing
    parts = _sd_target_parts(f)
    fam isa Poisson || block.label !== nothing || length(parts) == 1 || return nothing
    any(t -> t.target.level === :iid, parts) || return nothing
    return block
end

function _drm_count_ordinary_q2_sd(f::DrmFormula, fam::Union{Poisson,NegBinomial2}, block;
        data, tree, K, A, Ainv, coords, g_tol, se, nAGQ = nothing, plain::Bool = false)
    fname = fam isa Poisson ? "Poisson" : "NegBinomial2"
    group, slope, label = block.group, block.slope, block.label
    term = label === nothing ? "(1 + $slope | $group)" : "(1 + $slope | $label | $group)"
    who = "$fname `$term` SD regression"
    (tree === nothing && K === nothing && A === nothing && Ainv === nothing && coords === nothing) ||
        throw(ArgumentError("$who is an ordinary (non-structured) route; do not supply `tree`, `K`, `A`, `Ainv` or `coords`."))
    nAGQ === nothing || throw(ArgumentError("nAGQ is not used by the Laplace ordinary q2 SD route; omit it."))
    targets = _sd_target_parts(f)
    if !plain      # `plain`: `marginal = :Laplace` on `(1 + x | g)` with no SD formulas (constant SDs)
        length(targets) in (1, 2) || throw(ArgumentError(
            "$who needs an SD formula for the intercept, for the slope, or for both " *
            "(`sd($group, coef(intercept)) ~ …` and `sd($group, coef($slope)) ~ …`)."))
        all(item -> item.target.level === :iid && item.target.group === group &&
            item.target.dpar === :mu && item.target.block === nothing, targets) ||
            throw(ArgumentError("$who requires ordinary `sd($group, coef(…))` mean targets."))
        chosen = [item.target.member for item in targets]
        (all(m -> m in (:intercept, slope), chosen) && length(unique(chosen)) == length(chosen)) ||
            throw(ArgumentError("$who: SD formulas must select `coef(intercept)` and `coef($slope)`, each at most once."))
    end
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "$who cannot be combined with an intercept-only `sd($group, phylogenetic) ~ …` formula."))
    rhs = Dict(f.forms)
    # The block is the only random-effect term: rebuild the mean without it.
    terms = rhs[:mu] isa Tuple ? collect(rhs[:mu]) : Any[rhs[:mu]]
    rest = [t for t in terms if t !== block.term]
    rest_rhs = isempty(rest) ? ConstantTerm(1) : length(rest) == 1 ? rest[1] : Tuple(rest)
    fixed_mu, re, metav, st = _split_ranef(rest_rhs)
    (isempty(re) && metav === nothing && st === nothing) || throw(ArgumentError(
        "$who requires the `$term` block to be the only random-effect term."))
    (haskey(rhs, :zi) || haskey(rhs, :hu)) && throw(ArgumentError(
        "$who cannot yet be combined with `zi` or `hu`."))
    fixed_mu, offset_term = _extract_offset(fixed_mu)
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "$who supports only `mu`$(fam isa NegBinomial2 ? ", fixed `sigma ~ 1`," : "") and its two SD formulas."))
    all(_formula_response_observed_mask(f, data)) || throw(ArgumentError(
        "$who requires observed responses; remove incomplete rows before fitting."))
    y, Xmu, nmmu = _design(f.response, fixed_mu, data)
    all(yi -> isfinite(yi) && yi >= 0 && isinteger(yi), y) || throw(ArgumentError(
        "$fname() requires non-negative integer counts as the response"))
    sigma_rhs = get(rhs, :sigma, nothing)
    if fam isa NegBinomial2
        sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1) || throw(ArgumentError(
            "$who requires a fixed `sigma ~ 1` submodel."))
        offset_term === nothing || throw(ArgumentError(
            "NegBinomial2 random-effect SD regression does not yet support `offset(...)`."))
    else
        (sigma_rhs === nothing || (sigma_rhs isa ConstantTerm && sigma_rhs == ConstantTerm(1))) ||
            throw(ArgumentError("Poisson has no `sigma` parameter; remove the `sigma` formula."))
    end
    hasproperty(data, group) || throw(ArgumentError("$who: grouping column `$group` is absent from data."))
    gidx, G = _group_index(getproperty(data, group); col = group)
    Q = sparse(1.0I, G, G)
    # A coefficient with no SD formula keeps a constant SD (`sd(…, coef(·)) ~ 1`), as drmTMB does.
    # With `plain` there are no SD formulas: the loop is skipped and constant SDs are set below.
    ordered = plain ? Any[] : map((:intercept, slope)) do member
        hit = filter(item -> item.target.member === member, targets)
        isempty(hit) ? member : only(hit)
    end
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    blocks = Symbol[]
    for item in ordered
        if item isa Symbol
            push!(Zgs, ones(G, 1))
            push!(nmsds, ["(Intercept)"])
            push!(blocks, _sd_target_block((group = group, level = :iid, member = item,
                                            dpar = :mu, block = nothing); disambiguate = true))
            continue
        end
        Zg, names = _sd_group_design(f.response, item.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = item.key)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "$who: SD target `coef($(item.target.member))` has a rank-deficient group-level design."))
        push!(Zgs, Matrix{Float64}(Zg))
        push!(nmsds, names)
        push!(blocks, _sd_target_block(item.target; disambiguate = true))
    end
    if plain          # constant SDs, the same model as `sd(.) ~ 1` on both coefficients
        Zgs = [ones(G, 1), ones(G, 1)]
        nmsds = [["$(group):log_sd_intercept"], ["$(group):log_sd_$(slope)"]]
        blocks = [:sd_intercept, :sd_slope]
    end
    x = _poisson_animal_slope_loading(data, slope, length(y); label = "$fname ordinary")
    if all(==(first(x)), x) && isequal(Zgs[1], Zgs[2])
        throw(ArgumentError("$who cannot separate the intercept and slope SDs when " *
            "`$slope` is constant and both SD formulas have identical group-level designs."))
    end
    fit = if fam isa Poisson
        off = _poisson_sd_offset(offset_term, data, length(y))
        _fit_poisson_animal_intercept_slope_laplace(fam, y, Xmu, Zgs, gidx, Q, nmmu,
            nmsds, blocks, group, x, off, g_tol; se, correlated = true)
    else
        _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
        _fit_nb2_animal_intercept_slope_laplace(fam, y, Xmu, Xsigma, Zgs, gidx, Q,
            nmmu, nmsigma, nmsds, blocks, group, x, g_tol; se, correlated = true)
    end
    return _withformula(fit, f)
end
