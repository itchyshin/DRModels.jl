# Phylogenetic and coordinate-spatial independent intercept-plus-slope q2 blocks
# with one SD formula per coefficient:
#   y ~ x + phylo(1 + x | g)   with sd(g, phylogenetic, coef(intercept)) ~ z0,
#                                   sd(g, phylogenetic, coef(x)) ~ z1
#   y ~ x + spatial(1 + x | g) with sd(g, spatial, coef(intercept)) ~ z0,
#                                   sd(g, spatial, coef(x)) ~ z1
# The model is the animal/relmat q2 model with the kernel swapped:
#   eta_i = X_i b + s0_g f0_g + x_i s1_g f1_g,  f0, f1 iid N(0, C),
#   s_k = exp(Z_k alpha_k),  rho fixed at 0.
# Kernels match native drmTMB (checked against its log-likelihood on a
# height-2 ultrametric tree and an 8 x 8 grid):
#   phylo   C = tip correlation of the Brownian tree covariance (ape::vcv(tree,
#           corr = TRUE)); rows map to tips by name. Non-ultrametric trees are
#           refused, because drmTMB's correlation scale requires ultrametricity.
#   spatial C = exp(-d / median positive distance) + 1e-6 I, range FIXED (the
#           native `drm_spatial_coords_precision` kernel); rows of `coords` are
#           the group levels in first-seen order (after label resolution).
# The mesh/SPDE spatial route is not covered here.

# Return `(C, gidx, G)` for the phylo or coordinate-spatial q2 kernel.
function _structured_q2_kernel(kind::Symbol, grp::Symbol, labels; tree, K, A, coords,
                               who::AbstractString)
    if kind === :phylo
        (K === nothing && A === nothing && coords === nothing) || throw(ArgumentError(
            "$who uses `tree = …` only; do not also supply `K`, `A`, or `coords`."))
        tree === nothing && throw(ArgumentError("phylo(1 + x | $grp) needs `tree = …`"))
        phy = tree isa AbstractString ? augmented_phy(tree) : tree
        G = phy.n_leaves
        gidx = _phylo_mean_leaf_index(phy, labels)
        length(unique(gidx)) == G || throw(ArgumentError(
            "$who needs at least one row for every tip of `tree`; prune unobserved tips first."))
        S = sigma_phy_dense(phy; σ²_phy = 1.0)
        dS = diag(S)
        maximum(dS) - minimum(dS) <= 1e-8 * maximum(dS) || throw(ArgumentError(
            "$who uses the phylogenetic correlation scale, which requires an ultrametric " *
            "tree (as native drmTMB does); the root-to-tip depths here differ."))
        return _phylo_correlation(phy), gidx, G
    elseif kind === :spatial
        (K === nothing && A === nothing && tree === nothing) || throw(ArgumentError(
            "$who uses `coords = …` only; do not also supply `K`, `A`, or `tree`."))
        coords === nothing && throw(ArgumentError("spatial(1 + x | $grp) needs `coords = …`"))
        gidx, G = _group_index(labels)
        return _gaussian_spatial_sd_kernel(coords, G, grp), gidx, G
    end
    throw(ArgumentError("internal: unknown q2 kernel kind `$kind`"))
end

_q2_target_level(kind::Symbol) = kind === :phylo ? :phylogenetic : kind
_q2_marker(kind::Symbol) = kind === :phylo ? phylo : kind === :spatial ? spatial :
    kind === :animal ? animal : relmat

# Detect a phylo/spatial `(1 + x | g)` block that carries a matching coefficient
# SD target. Returns `(kind, block)` or `nothing`.
function _phylo_spatial_q2_block(f::DrmFormula)
    rhs = get(Dict(f.forms), :mu, nothing)
    rhs === nothing && return nothing
    targets = _sd_target_parts(f)
    for kind in (:phylo, :spatial)
        block = _animal_intercept_slope(rhs; kind = _q2_marker(kind))
        block === nothing && continue
        any(item -> item.target.level === _q2_target_level(kind), targets) &&
            return (kind, block)
    end
    return nothing
end

# Poisson and NegBinomial2 phylo/spatial q2 (sparse-wrapped dense precision
# C^{-1}, same Laplace fit helpers as the animal/relmat q2 routes).
function _drm_count_phylo_spatial_q2_sd(f::DrmFormula, fam::Union{Poisson,NegBinomial2},
        kind::Symbol, block; data, tree, K, A, Ainv, coords, g_tol, se, nAGQ = nothing)
    fname = fam isa Poisson ? "Poisson" : "NegBinomial2"
    level = _q2_target_level(kind)
    group, slope = block.group, block.slope
    term = "$kind(1 + $slope | $group)"
    who = "$fname `$term` SD regression"
    targets = _sd_target_parts(f)
    length(targets) == 2 || throw(ArgumentError(
        "$who needs one SD formula for the intercept and one for the slope " *
        "(`sd($group, $level, coef(intercept)) ~ …` and `sd($group, $level, coef($slope)) ~ …`)."))
    all(item -> item.target.level === level && item.target.group === group &&
        item.target.dpar === :mu && item.target.block === nothing, targets) ||
        throw(ArgumentError("$who requires two matching `$level` mean targets."))
    Set(item.target.member for item in targets) == Set((:intercept, slope)) ||
        throw(ArgumentError("$who: SD formulas must select `coef(intercept)` and `coef($slope)` exactly once."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "$who cannot be combined with an intercept-only `sd($group, phylogenetic) ~ …` formula."))
    Ainv === nothing || throw(ArgumentError("$who does not use `Ainv`."))
    nAGQ === nothing || throw(ArgumentError("nAGQ is not used by the Laplace $kind q2 SD route; omit it."))
    rhs = Dict(f.forms)
    fixed_mu, re, metav, structured, phylo_slope, _, other_slope = _split_ranef(rhs[:mu];
        allow_phylo_slope = kind === :phylo,
        allow_spatial_intercept_slope = kind === :spatial)
    parsed_slope = kind === :phylo ? phylo_slope : other_slope
    metav === nothing && isempty(re) && structured == (kind, group) &&
        parsed_slope === slope || throw(ArgumentError(
        "$who requires one matching independent `$term` term and no ordinary random effects or `meta_V`."))
    _collect_structured(rhs[:mu]) == [(kind, group)] || throw(ArgumentError(
        "$who requires exactly one structured term, with no additional structured effects."))
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
    C, gidx, G = _structured_q2_kernel(kind, group, getproperty(data, group);
                                       tree, K, A, coords, who)
    Cchol = cholesky(Symmetric(C); check = false)
    issuccess(Cchol) || throw(ArgumentError("$who: the $kind kernel is not positive definite."))
    Q = sparse(Symmetric(inv(Cchol)))
    ordered = map((:intercept, slope)) do member
        only(filter(item -> item.target.member === member, targets))
    end
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    for item in ordered
        Zg, names = _sd_group_design(f.response, item.rhs, data, gidx, G, group;
            schema_cache = f.schema_cache, schema_key = item.key)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "$who: SD target `coef($(item.target.member))` has a rank-deficient group-level design."))
        push!(Zgs, Matrix{Float64}(Zg))
        push!(nmsds, names)
    end
    x = _poisson_animal_slope_loading(data, slope, length(y); label = "$fname $kind")
    if all(==(first(x)), x) && isequal(Zgs[1], Zgs[2])
        throw(ArgumentError("$who cannot separate the intercept and slope variances when " *
            "`$slope` is constant and both SD formulas have identical group-level designs."))
    end
    blocks = Symbol[_sd_target_block(item.target; disambiguate = true) for item in ordered]
    fit = if fam isa Poisson
        off = _poisson_sd_offset(offset_term, data, length(y))
        _fit_poisson_animal_intercept_slope_laplace(fam, y, Xmu, Zgs, gidx, Q, nmmu,
            nmsds, blocks, group, x, off, g_tol; se)
    else
        _, Xsigma, nmsigma = _design(f.response, sigma_rhs, data)
        _fit_nb2_animal_intercept_slope_laplace(fam, y, Xmu, Xsigma, Zgs, gidx, Q,
            nmmu, nmsigma, nmsds, blocks, group, x, g_tol; se)
    end
    return _withformula(fit, f)
end
