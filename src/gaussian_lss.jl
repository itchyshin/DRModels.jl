# Location–scale–scale (#544): a third submodel putting a linear predictor on the
# LOG standard deviation of the (1 | g) random effect, mirroring drmTMB's
# `bf(y ~ x + (1 | g), sigma ~ x, sd(g) ~ x_group)` grammar (vignette
# "location-scale-scale"). Per group k: σ_b,k = exp(Z_k' α), so the marginal is
# still the exact Woodbury Gaussian of `_fit_ranef_gaussian` — the capacitance is
# diagonal per group, so a per-group variance is a drop-in. The scalar model is
# the special case Zg = ones(G, 1); `test/test_lss_group.jl` pins that reduction.
#
# `sd_phylo(species) ~ x` (#545, the Mizuno et al. QQQ model's phylogenetic
# scale component) lives in the second half of this file: dense scaled-structure
# engine first (correctness), O(p) sparse to follow (performance).

"""
    sd(group)

Formula marker for a location–scale–scale model. Used only on the left-hand
side of a `bf` component formula,

```julia
bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x), @formula(sd(g) ~ z))                  # iid RE SD
bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x), @formula(sd(species, phylogenetic) ~ z))  # phylo SD
bf(@formula(cbind(s, f) ~ x), @formula(sigma ~ 1 + (1 | g)), @formula(sd(g, sigma) ~ z))  # BetaBinomial sigma RE SD
```

to put a linear predictor on the **log standard deviation of a random
effect**. Plain `sd(g)` targets the iid `(1 | g)` intercept
(`log σ_b,k = Z_k' α`, one row per group level); `sd(group, phylogenetic)`
targets the per-species phylogenetic SD (drmTMB:
`sd(group, level = "phylogenetic")`; `@formula` cannot parse keyword
arguments, so the level is a bare symbol). Predictors must be constant
within each group level. The positional `sd(g, sigma)` selector targets an
ordinary random intercept on the BetaBinomial `sigma` predictor; only this
family route currently supports a non-mean target. Coefficients appear in the
`:sd` (iid) or `:sd_phylo` (phylogenetic) block.

## Capabilities

- **Estimators**: supports both Maximum Likelihood (`method = :ML`, default)
  and Restricted Maximum Likelihood (`method = :REML`).
- **Missing response**: incomplete responses (`missing` or `NaN` in `y`) are
  fit via the observed-rows pattern (`response = "include"` in R bridge),
  preserving group and phylogenetic structures across all G levels.
- **Sparse scaling**: phylogenetic LSS models scale to large trees in O(p) time
  via `sparse = true` (or `algorithm = :sparse_lbfgs`), automatically selected
  when G > 500 species.
- **Multi-component models**: combines multiple grouping factors or iid + phylogenetic
  random-effect SD models.
"""
sd(x) = x

# SD formulas use private unique keys so multiple coefficients on one group can
# coexist. Retain a fallback for formula objects built by older constructors.
function _sd_target_parts(f::DrmFormula)
    out = NamedTuple[]
    for p in f.forms
        key = first(p)
        if haskey(f.sd_targets, key)
            push!(out, (; key, target = f.sd_targets[key], rhs = last(p)))
        elseif startswith(String(key), "sd_") && !startswith(String(key), "sdphy_")
            grp = Symbol(String(key)[(length("sd_")+1):end])
            push!(out, (; key, target = (group=grp, level=:iid, member=nothing,
                                         dpar=:mu, block=nothing), rhs=last(p)))
        end
    end
    out
end

_sd_parts(f::DrmFormula) = Pair{Symbol,Any}[
    item.target.group => item.rhs for item in _sd_target_parts(f)
]

# Parameter block names for coefficient-selected SD targets are derived from
# the complete target descriptor, not formula order or a shared group prefix.
function _sd_target_block(target::NamedTuple; disambiguate::Bool = false)
    target.member === nothing && !disambiguate && return :sd
    block = target.block === nothing ? "none" : String(target.block)
    member = target.member === nothing ? "none" : String(target.member)
    identity = join((String(target.group), String(target.level), String(target.dpar),
                     member, block), "|")
    Symbol("sdtarget_", bytes2hex(codeunits(identity)))
end

_sd_target_blocks(targets) = [
    _sd_target_block(item.target; disambiguate = length(targets) > 1)
    for item in targets
]

# One-line guard for family routes that do not support sd(): every family
# consumer reads `Dict(f.forms)` by known keys, so an unguarded sd() part would
# be DROPPED SILENTLY — the exact bug class of issue #2 (silent σ-phylo drop).
function _lss_only_gaussian_guard(f::DrmFormula, fam)
    (isempty(_sd_parts(f)) && isempty(_sdphylo_parts(f))) ||
        throw(ArgumentError("drm: random-effect `sd(group) ~ …` formulas are currently " *
            "supported for Gaussian LSS models and narrow ordinary IID routes in " *
            "Binomial, Poisson, NegBinomial2, Gamma, and Beta (got $(nameof(typeof(fam))))."))
    return nothing
end

# Group-level design for the sd() submodel: build the n-row design from the full
# data, require every predictor constant within each group level (drmTMB errors
# the same way), and compress to one row per group in the supplied `gidx` order.
# IID callers use `_group_index`'s first-seen order; phylogenetic callers supply
# the tree-tip order through `_lss_phylo_group_index` below.
function _sd_group_design(response::Symbol, rhs, data, gidx::Vector{Int}, G::Int, grp::Symbol;
                          schema_cache = nothing, schema_key = nothing,
                          store_schema::Bool = true)
    _, Zfull, nm = _design(response, rhs, data; schema_cache, schema_key, store_schema)
    q = size(Zfull, 2)
    Zg = fill(NaN, G, q)
    for i in eachindex(gidx)
        k = gidx[i]
        if isnan(Zg[k, 1])
            @views Zg[k, :] .= Zfull[i, :]
        else
            for j in 1:q
                Zg[k, j] == Zfull[i, j] ||
                    throw(ArgumentError("sd($grp): predictor `$(nm[j])` varies within a level " *
                        "of `$grp` — sd() predictors must be constant within each group. " *
                        "Check the grouping variable and source data; do not average a " *
                        "genuinely within-group predictor to silence this."))
            end
        end
    end
    return Zg, nm
end

# Router leg: validate the sd() request against the rest of the Gaussian model
# and dispatch. Called from `drm(f, ::Gaussian)` with everything it has parsed.
function _drm_gaussian_lss(f::DrmFormula, fam::Gaussian, sdp, re, re_kinds, structured,
                           structured_sigma, sigma_re, metav, has_missing_response,
                           y, Xμ, Xσ, nmμ, nmσ, data, g_tol, method)
    length(sdp) == 1 ||
        throw(ArgumentError("drm: one `sd(group) ~ …` formula per model (got $(length(sdp)))."))
    sgrp, srhs = sdp[1]
    sdtargets = _sd_target_parts(f)
    length(sdtargets) == 1 || throw(ArgumentError(
        "drm: Gaussian LSS requires exactly one resolved random-effect SD target."))
    descriptor = only(sdtargets).target
    (descriptor.level === :iid && descriptor.dpar === :mu &&
     descriptor.block === nothing && descriptor.group === sgrp) ||
        throw(ArgumentError("drm: Gaussian `sd($sgrp)` must target one ordinary mean random-effect coefficient."))
    (structured === nothing && structured_sigma === nothing) ||
        throw(ArgumentError("drm: `sd($sgrp)` with structured effects (phylo/relmat/animal/" *
            "spatial) is not yet supported — the climate-dependent phylogenetic SD " *
            "(`sd_phylo`) is issue #545."))
    metav === nothing ||
        throw(ArgumentError("drm: `sd($sgrp)` cannot be combined with `meta_V(...)`."))
    isempty(sigma_re) ||
        throw(ArgumentError("drm: `sd($sgrp)` cannot be combined with a random effect on `sigma`."))
    length(re) == 1 ||
        throw(ArgumentError("drm: `sd($sgrp)` requires exactly one ordinary mean random-effect term " *
            "(got $(length(re)) random-effect terms)."))
    (kind, slope) = re_kinds[1]
    kind in (:intercept, :slope) ||
        throw(ArgumentError("drm: Gaussian `sd($sgrp)` supports one ordinary intercept or one independent numeric slope."))
    member = kind === :intercept ? :intercept : slope
    descriptor.member in (nothing, member) ||
        throw(ArgumentError("drm: Gaussian SD target `coef($(descriptor.member))` does not match mean coefficient `$(member)`."))
    (kind === :intercept || descriptor.member === member) ||
        throw(ArgumentError("drm: Gaussian random slopes require an explicit `coef($slope)` SD target."))
    (_, grp) = re[1]
    grp === sgrp ||
        throw(ArgumentError("drm: `sd($sgrp)` must name the grouping of the random effect — " *
            "the mean formula has `(1 | $grp)`."))
    gidx, G = _group_index(getproperty(data, grp); col = grp)
    target_key = only(sdtargets).key
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = target_key)
    loading = if kind === :intercept
        ones(Float64, length(y))
    else
        hasproperty(data, slope) || throw(ArgumentError(
            "drm: Gaussian random-slope predictor `$slope` is absent from data."))
        raw = getproperty(data, slope)
        raw isa AbstractVector{<:Real} && length(raw) == length(y) && all(isfinite, raw) ||
            throw(ArgumentError("drm: Gaussian random-slope predictor `$slope` must be finite numeric values, one per response."))
        any(x -> !iszero(x), raw) || throw(ArgumentError(
            "drm: Gaussian random-slope predictor `$slope` must contain at least one nonzero value."))
        Float64.(raw)
    end
    if has_missing_response
        observed = _observed_response_mask(y)
        n_obs = count(observed)
        pμ, pσ, psd = size(Xμ, 2), size(Xσ, 2), size(Zg, 2)
        total_dof = pμ + pσ + psd
        n_obs >= total_dof ||
            throw(ArgumentError("drm (Gaussian sd): only $(n_obs) observed responses for a model with " *
                "$(total_dof) parameters ($(pμ) mean + $(pσ) scale + $(psd) sd) — " *
                "too few to fit. Use `drm_listwise` or supply more complete responses."))
        n_obs > total_dof ||
            @warn "drm (Gaussian sd): $(n_obs) observed responses equals the $(total_dof) model " *
                  "parameters (residual dof 0); the fit is saturated and inference is unreliable."
        @warn "drm: $(length(observed) - n_obs) of $(length(observed)) rows have a missing/NaN " *
              "response and were dropped (observed-rows fit, like glmmTMB's default na.action). " *
              "Use `drm_listwise` to preprocess explicitly, or supply complete responses, to silence this."
        y = Vector{Float64}(y[observed])
        Xμ = Matrix{Float64}(Xμ[observed, :])
        Xσ = Matrix{Float64}(Xσ[observed, :])
        gidx = gidx[observed]
        loading = loading[observed]
    end
    return _fit_ranef_gaussian_lss(fam, y, Xμ, Xσ, Zg, gidx, G, nmμ, nmσ, nmsd, grp, g_tol;
                                   reml = method === :REML, w = loading)
end

# Gaussian location–scale–scale with one mean random intercept (1 | g) whose
# per-group SD carries a linear predictor: θ = [β_μ; β_σ (log σ); α (log σ_b)].
# Same exact Woodbury/Patterson–Thompson marginal as `_fit_ranef_gaussian`
# (which is left byte-for-byte untouched), with σ_b² per group and optional
# observation-level coefficient loadings for independent random slopes. Convergence
# reasoning (exact marginal, no inner-solve noise floor) carries over from that
# function's header note.
function _fit_ranef_gaussian_lss(fam::Gaussian, y, Xμ, Xσ, Zg, gidx, G, nmμ, nmσ, nmsd,
                                 grp, g_tol; reml::Bool = false, w = nothing)
    n = length(y)
    pμ, pσ, psd = size(Xμ, 2), size(Xσ, 2), size(Zg, 2)
    const_2pi = 0.5 * n * log(2π)
    const_pμ = 0.5 * pμ * log(2π)

    function nll_ml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; α = θ[pμ+pσ+1:pμ+pσ+psd]
        ημ = Xμ * βμ; ησ = Xσ * βσ                 # ησ = log σ_i
        ησb = Zg * α                                # per-group log σ_b,k
        T = eltype(θ)
        S = zeros(T, G); C = zeros(T, G)
        rv = Vector{T}(undef, n); invDv = Vector{T}(undef, n)
        logdetD = zero(T)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i])
            r = y[i] - ημ[i]
            rv[i] = r; invDv[i] = invD
            k = gidx[i]
            wi = w === nothing ? one(T) : w[i]
            S[k] += wi * wi * invD
            C[k] += wi * r * invD
            logdetD += 2 * ησ[i]
        end
        logdetCap = zero(T)
        @inbounds for k in 1:G
            σb² = exp(2 * ησb[k])
            logdetCap += log(1 + σb² * S[k])
        end
        # Cancellation-free r′V⁻¹r (#746/#747; see `_re_quad_stable`).
        quad = _re_quad_stable(rv, invDv, w, gidx, exp.(-2 .* ησb), S, C)
        return 0.5 * (logdetD + logdetCap + quad) + const_2pi
    end

    function nll_reml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; α = θ[pμ+pσ+1:pμ+pσ+psd]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        ησb = Zg * α
        T = eltype(θ)
        S = zeros(T, G)
        invDv = Vector{T}(undef, n)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i])
            invDv[i] = invD
            wi = w === nothing ? one(T) : w[i]
            S[gidx[i]] += wi * wi * invD
        end
        # PSD penalised-SS form of Xμ′V⁻¹Xμ. The Woodbury subtraction it replaces
        # returned exactly-singular garbage (8e110 vs a true 1e-7) at an LBFGS
        # probe with log σ_i ≈ -105; see `_re_xtvinvx_stable` and
        # test/test_lss_reml_falseconv.jl.
        XtVinvX = _re_xtvinvx_stable(Xμ, invDv, w, gidx, exp.(-2 .* ησb), S)
        # ML part via the cancellation-free `nll_ml` (#746/#747).
        nll_ml_θ = nll_ml(θ)
        # Same Woodbury-subtraction PSD hazard as `_fit_ranef_gaussian.nll_reml`
        # (#499): reject a non-PD Xμ′V⁻¹Xμ with a large FINITE barrier.
        # The generic (ForwardDiff Dual) Cholesky ACCEPTS a zero pivot, so also
        # reject a non-finite logdet: -Inf there is what broke HagerZhang.
        cholXtVinvX = cholesky(Symmetric(XtVinvX); check=false)
        issuccess(cholXtVinvX) || return nll_ml_θ + T(REML_NONPD_PENALTY)
        ldX = logdet(cholXtVinvX)
        isfinite(ldX) || return nll_ml_θ + T(REML_NONPD_PENALTY)
        return nll_ml_θ + 0.5 * ldX - const_pμ
    end

    nll = reml ? nll_reml : nll_ml

    βμ0 = Xμ \ y
    res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + psd)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) + eps())
    # Start the sd() submodel at the constant that matches the scalar route's
    # init: solve Zg α ≈ log(std(res0)/2) in least squares (robust to any
    # column coding; with Zg = ones(G,1) this is exactly the scalar init).
    θ0[pμ+pσ+1:end] .= Zg \ fill(log(std(res0) / 2 + eps()), G)

    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    H = ForwardDiff.hessian(nll, θ̂)
    V = _vcov_from_hessian(H)
    # Absolute `g_tol` is not unit-free: a scaled predictor stalls just above 1e-8.
    # Neither `g_converged` (absolute 1e-8) nor the x/f/g OR `Optim.converged`
    # is part of this flag (#983). Stopping on the iteration cap is not a minimum.
    # A non-finite Hessian is not converged: `_unitfree_converged` returns false
    # there (#1114, #983).
    converged = !Optim.iteration_limit_reached(res) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :sd => (pμ+pσ+1):(pμ+pσ+psd)]
    names = [:mu => nmμ, :sigma => nmσ, :sd => nmsd]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    sd_group = exp.(Zg * θ̂[(pμ+pσ+1):(pμ+pσ+psd)])
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
                  :sd_group => sd_group, :sd => sd_group[gidx])
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]; α = θ̂[pμ+pσ+1:end]
        ημ = Xμ * βμ; ησ = Xσ * βσ; ησb = Zg * α
        S = zeros(G); C = zeros(G)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i]); k = gidx[i]
            wi = w === nothing ? 1.0 : w[i]
            S[k] += wi * wi * invD
            C[k] += wi * (y[i] - ημ[i]) * invD
        end
        [C[k] / (1 / exp(2 * ησb[k]) + S[k]) for k in 1:G]
    end
    re = Dict(Symbol(grp) => blup)
    fit = _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n,
                                     converged, means, obs, scales), nll_ml), re)
    if reml
        return _withreml(fit, -nll_reml(θ̂), -nll_ml(θ̂))
    end
    return fit
end

# ---------------------------------------------------------------------------
# #545 — sd_phylo(species) ~ x: climate-dependent PHYLOGENETIC SD (the Mizuno
# et al. QQQ model's phylogenetic scale component). Marginal, per Eq. 25 of the
# registered report:  V = D_a A D_a + D_e²,  log σ_a,k = Z_k' α  (one row per
# species), log σ_e,i = Xσ_i' βσ (the existing heteroscedastic residual).
# Dense-Woodbury engine (correctness first; O(p) sparse is the perf follow-up):
# Σ_a = D_a K D_a ⇒ Σ_a⁻¹ = D_a⁻¹ K⁻¹ D_a⁻¹, logdet Σ_a = logdet K + 2 Σ log σ_a,k.

"""
    sd_phylo(group)

DEPRECATED legacy spelling for the phylogenetic SD submodel — use
`sd(group, phylogenetic) ~ …` (drmTMB: `sd(group, level = "phylogenetic")`).
Both put a linear predictor on the **log per-species SD of the phylogenetic
random effect**: `a ~ MVN(0, D_a K D_a)` with `D_a = Diagonal(exp.(Z * α))`
and `K` the Brownian-motion phylogenetic correlation from `tree`. Kept
working, exactly as the twin keeps its legacy spelling, and canonicalised to
the same `:sd_phylo` block; new code should write the `sd(group, …)` form.
"""
sd_phylo(x) = x

_sdphylo_parts(f::DrmFormula) = Pair{Symbol,Any}[
    Symbol(String(first(p))[(length("sdphy_")+1):end]) => last(p)
    for p in f.forms if startswith(String(first(p)), "sdphy_")]

"""
    _lss_phylo_group_index(tree, labels, grp) -> phy, gidx, G

Resolve a Gaussian LSS phylogenetic grouping column against its tree.  Integer
labels are positional leaf indices `1:p`; text labels match `phy.leaf_names`
exactly.  The full input must contain every leaf before any missing response is
removed, so an all-missing response tip remains a prior-only tree state instead
of changing the covariance dimension.  This helper deliberately accepts only
the shipped `AugmentedPhy` and Newick-string tree contracts; arbitrary
`sigma_phy_dense` providers are not qualified for labelled LSS identity.
"""
function _lss_phylo_group_index(tree, labels, grp::Symbol)
    phy = if tree isa AugmentedPhy
        tree
    elseif tree isa AbstractString
        augmented_phy(tree)
    else
        throw(ArgumentError("drm: phylo LSS `$grp` requires an AugmentedPhy or Newick string tree"))
    end
    values = collect(labels)
    isempty(values) && throw(ArgumentError("drm: phylo LSS grouping `$grp` is empty"))
    any(ismissing, values) &&
        throw(ArgumentError("drm: phylo LSS grouping `$grp` contains a missing tree-tip label"))
    p = phy.n_leaves
    gidx = if all(value -> value isa Integer, values)
        all(value -> 1 <= value <= p, values) ||
            throw(ArgumentError("drm: integer phylo LSS labels for `$grp` must be positional tips in 1:$p"))
        Int.(values)
    elseif all(value -> applicable(String, value), values)
        by_name = Dict(name => i for (i, name) in enumerate(phy.leaf_names))
        index = Vector{Int}(undef, length(values))
        for (row, value) in enumerate(values)
            name = String(value)
            haskey(by_name, name) ||
                throw(ArgumentError("drm: phylo LSS group label `$name` for `$grp` is not a tree tip name"))
            index[row] = by_name[name]
        end
        index
    else
        throw(ArgumentError("drm: phylo LSS grouping `$grp` must use all integer tip positions or text values with exact tree-tip names"))
    end
    present = falses(p)
    present[gidx] .= true
    if !all(present)
        absent = join(phy.leaf_names[findall(.!present)], ", ")
        throw(ArgumentError("drm: phylo LSS full input for `$grp` must contain all tree tips; absent: $absent. " *
            "The tree has tips with no row in the data; prune the tree to the species in the data " *
            "(for example `ape::keep.tip(tree, observed_species)` in R) before fitting."))
    end
    return phy, gidx, p
end

# Router leg for sd_phylo: validate against the parsed Gaussian model and
# dispatch the dense scaled-structure fitter.
function _drm_gaussian_lss_phylo(f::DrmFormula, fam::Gaussian, sdpp, re, structured,
                                 structured_sigma, sigma_re, metav, has_missing_response,
                                 y, Xμ, Xσ, nmμ, nmσ, data, tree, g_tol, method, penalty;
                                 algorithm::Symbol = :auto, sparse = nothing)
    length(sdpp) == 1 ||
        throw(ArgumentError("drm: one `sd_phylo(group) ~ …` formula per model (got $(length(sdpp)))."))
    sgrp, srhs = sdpp[1]
    isempty(_sd_parts(f)) ||
        throw(ArgumentError("drm: `sd_phylo($sgrp)` cannot be combined with a plain `sd(group)` " *
            "formula yet — one scale–scale submodel per fit."))
    structured !== nothing ||
        throw(ArgumentError("drm: `sd_phylo($sgrp)` requires a `phylo(1 | $sgrp)` structured " *
            "random effect in the mean formula."))
    kind, grp = structured
    kind === :phylo ||
        throw(ArgumentError("drm: `sd_phylo($sgrp)` requires a `phylo(1 | g)` mean random effect " *
            "(the mean has `$kind(1 | $grp)`); sd() submodels for relmat/animal are a follow-up."))
    grp === sgrp ||
        throw(ArgumentError("drm: `sd_phylo($sgrp)` must name the phylo grouping — the mean " *
            "formula has `phylo(1 | $grp)`."))
    structured_sigma === nothing ||
        throw(ArgumentError("drm: `sd_phylo($sgrp)` with a σ-phylo random effect is not supported — " *
            "the residual scale takes FIXED-effect predictors (`sigma ~ x`) in this route."))
    metav === nothing ||
        throw(ArgumentError("drm: `sd_phylo($sgrp)` cannot be combined with `meta_V(...)`."))
    (isempty(re) && isempty(sigma_re)) ||
        throw(ArgumentError("drm: `sd_phylo($sgrp)` supports no additional random effects " *
            "beyond the phylo structured intercept."))
    penalty === nothing ||
        throw(ArgumentError("drm: `penalty` is not wired for the sd_phylo route."))
    tree === nothing && error("phylo(1 | $sgrp) needs `tree = …`")
    phy, gidx, G = _lss_phylo_group_index(tree, getproperty(data, sgrp), sgrp)
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, sgrp)
    if has_missing_response
        observed = _observed_response_mask(y)
        n_obs = count(observed)
        pμ, pσ, psd = size(Xμ, 2), size(Xσ, 2), size(Zg, 2)
        total_dof = pμ + pσ + psd
        n_obs >= total_dof ||
            throw(ArgumentError("drm (Gaussian sd_phylo): only $(n_obs) observed responses for a model with " *
                "$(total_dof) parameters ($(pμ) mean + $(pσ) scale + $(psd) sd_phylo) — " *
                "too few to fit. Use `drm_listwise` or supply more complete responses."))
        n_obs > total_dof ||
            @warn "drm (Gaussian sd_phylo): $(n_obs) observed responses equals the $(total_dof) model " *
                  "parameters (residual dof 0); the fit is saturated and inference is unreliable."
        @warn "drm: $(length(observed) - n_obs) of $(length(observed)) rows have a missing/NaN " *
              "response and were dropped (observed-rows fit, like glmmTMB's default na.action). " *
              "Use `drm_listwise` to preprocess explicitly, or supply complete responses, to silence this."
        y = Vector{Float64}(y[observed])
        Xμ = Matrix{Float64}(Xμ[observed, :])
        Xσ = Matrix{Float64}(Xσ[observed, :])
        gidx = gidx[observed]
    end

    use_sparse = (sparse === true) || (algorithm in (:sparse, :sparse_lbfgs)) || (algorithm === :auto && G > 500)
    if use_sparse
        return _fit_phylo_gaussian_lss_sparse(fam, y, Xμ, Xσ, Zg, gidx, G, phy, nmμ, nmσ, nmsd,
                                              sgrp, g_tol; reml = method === :REML)
    else
        Kmat = _phylo_correlation(phy)
        size(Kmat) == (G, G) ||
            error("structured matrix must be $(G)×$(G) (the number of `$sgrp` levels)")
        return _fit_structured_gaussian_lss(fam, y, Xμ, Xσ, Zg, gidx, G, Kmat, nmμ, nmσ, nmsd,
                                            sgrp, g_tol; reml = method === :REML)
    end
end

# Marginal for the scaled-structure model, assembled DIRECTLY as
#     V = Z (D_a K D_a) Z' + diag(σ_e,i²)
# and factorised with a dense Cholesky. θ = [β_μ; β_σ (log σ_e); α (log σ_a)].
#
# WHY NOT WOODBURY HERE (measured 2026-08-28). The Woodbury/capacitance form used
# by `_fit_structured_gaussian` computes the quadratic as `q1 - dot(C, M \ C)`.
# With a per-species D_a the optimiser can walk into σ_e → 0, where q1 and the
# correction are BOTH enormous and their difference is pure rounding noise: the
# objective there returned nll ≈ −3.3e12 while the true dense value was +4.7e6,
# and LBFGS chased that artifact to a garbage optimum (σ_e ≈ e^−10). The
# likelihood itself was correct — verified against a brute-force dense evaluation
# at the true parameters to 1e-14 — so this is a CANCELLATION failure, not a
# derivation error. The dense assembly has no such subtraction.
#
# COST. O(n³) per evaluation, so this route is for species-level datasets up to a
# few thousand rows: the family- and order-level analyses of the Mizuno et al.
# protocol (N ≈ 50–500) fit instantly. The whole-tree scope (~10⁴ species) uses
# the sparse O(p) augmented-state spine, selected automatically by the public
# router above 500 species or explicitly with `sparse = true`.
function _fit_structured_gaussian_lss(fam::Gaussian, y, Xμ, Xσ, Zg, gidx, G, K,
                                      nmμ, nmσ, nmsd, grp, g_tol;
                                      block::Symbol = :sd_phylo, reml::Bool = false,
                                      loading = nothing, report_sd::Bool = false)
    n = length(y)
    # `loading` is the per-observation weight of the structured effect: `nothing`
    # is a structured intercept (weight 1); a numeric vector `x` makes it the
    # slope-only effect `x_i * u_g` with `u = D_a b`, `b ~ N(0, K)`.
    w = loading === nothing ? ones(Float64, n) : Float64.(loading)
    length(w) == n || throw(ArgumentError("structured loading must have one value per row"))
    pμ, pσ, psd = size(Xμ, 2), size(Xσ, 2), size(Zg, 2)
    const_2pi = 0.5 * n * log(2π)
    const_pμ = 0.5 * pμ * log(2π)
    n ≤ 5000 || throw(ArgumentError("drm: this forced dense `sd_phylo` route assembles a $(n)×$(n) " *
        "marginal covariance and is limited to 5000 rows. Use `sparse = true` or " *
        "`algorithm = :sparse_lbfgs` for the existing O(p) whole-tree engine."))
    Ksym = Symmetric(_finite_user_matrix(K, "K"))

    function nll_ml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; α = θ[pμ+pσ+1:pμ+pσ+psd]
        ημ = Xμ * βμ; ησ = Xσ * βσ; ησa = Zg * α
        T = eltype(θ)
        σa = exp.(ησa)                                    # per-species phylo SD
        Σa = (σa * σa') .* Ksym                           # D_a K D_a
        # NOT named `V` (#549): the enclosing fitter also assigns `V` (the vcov),
        # and a name assigned in BOTH a closure and its enclosing function is ONE
        # shared boxed variable in Julia. This closure is stored on the fit and
        # called CONCURRENTLY by threaded profile CIs, so the shared box was a
        # data race — measured as Dual-tag type mixing (two threads' matrices
        # crossing) and, from R, a HagerZhang `AssertionError: B > A`.
        Vm = Matrix{T}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            Vm[i, j] = w[i] * w[j] * Σa[gidx[i], gidx[j]]     # Z Σ_a Z'
        end
        @inbounds for i in 1:n
            Vm[i, i] += exp(2 * ησ[i])                    # + σ_e,i²
        end
        Vfac = cholesky(Symmetric(Vm); check = false)
        issuccess(Vfac) || return convert(T, 1e18)
        r = y .- ημ
        return 0.5 * (logdet(Vfac) + dot(r, Vfac \ r)) + const_2pi
    end

    function nll_reml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; α = θ[pμ+pσ+1:pμ+pσ+psd]
        ημ = Xμ * βμ; ησ = Xσ * βσ; ησa = Zg * α
        T = eltype(θ)
        σa = exp.(ησa)
        Σa = (σa * σa') .* Ksym
        Vm = Matrix{T}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            Vm[i, j] = w[i] * w[j] * Σa[gidx[i], gidx[j]]
        end
        @inbounds for i in 1:n
            Vm[i, i] += exp(2 * ησ[i])
        end
        Vfac = cholesky(Symmetric(Vm); check = false)
        issuccess(Vfac) || return convert(T, 1e18)
        r = y .- ημ
        nll_ml_θ = 0.5 * (logdet(Vfac) + dot(r, Vfac \ r)) + const_2pi
        A = Vfac \ Xμ
        XtVinvX = Xμ' * A
        chX = cholesky(Symmetric(XtVinvX); check = false)
        issuccess(chX) || return nll_ml_θ + convert(T, REML_NONPD_PENALTY)
        return nll_ml_θ + 0.5 * logdet(chX) - const_pμ
    end

    nll = reml ? nll_reml : nll_ml

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + psd)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) / sqrt(2) + eps())
    # A slope effect multiplies x, so its SD starts at the residual scale over rms(x).
    s0 = std(res0) / sqrt(2) / max(sqrt(sum(abs2, w) / n), eps())
    θ0[pμ+pσ+1:end] .= Zg \ fill(log(s0 + eps()), G)
    # NaN turns off the f- and x-stops, same as the temporal REML-style call.
    # A masked phylogenetic leaf can make two successive REML objectives
    # bitwise identical while |g|∞ is still above g_tol. The fit must finish
    # on the gradient.
    res = Optim.optimize(nll, θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = g_tol,
                                       x_abstol = NaN, x_reltol = NaN,
                                       f_abstol = NaN, f_reltol = NaN);
                         autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    H = ForwardDiff.hessian(nll, θ̂)
    V = _vcov_from_hessian(H)
    # Same flag as the scalar route: not the iteration cap, and the Newton step
    # in SE units. `g_converged` and the x/f/g OR stay out (#983). A non-finite
    # Hessian is not converged (#1114).
    converged = !Optim.iteration_limit_reached(res) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), block => (pμ+pσ+1):(pμ+pσ+psd)]
    names = [:mu => nmμ, :sigma => nmσ, block => nmsd]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    if report_sd
        sd_group = exp.(Zg * θ̂[(pμ+pσ+1):end])
        scales[:sd_group] = sd_group
        scales[:sd] = sd_group[gidx]
        scales[:sd_group_index] = Float64.(gidx)
    end
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]; α = θ̂[pμ+pσ+1:end]
        ημ = Xμ * βμ; ησ = Xσ * βσ; ησa = Zg * α
        σa = exp.(ησa)
        Σa = (σa * σa') .* Ksym
        Vm = Matrix{Float64}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            Vm[i, j] = w[i] * w[j] * Σa[gidx[i], gidx[j]]
        end
        @inbounds for i in 1:n
            Vm[i, i] += exp(2 * ησ[i])
        end
        Vfac = cholesky(Symmetric(Vm); check = false)
        r = y .- ημ
        Vinv_r = Vfac \ r
        ZtVinv_r = zeros(G)
        for i in 1:n
            ZtVinv_r[gidx[i]] += w[i] * Vinv_r[i]
        end
        Σa * ZtVinv_r
    end
    re = Dict(Symbol(grp) => blup)
    # b34: the structured SD route (relmat / animal / spatial; `report_sd`) also
    # records the conditional field for the R bridge. `blup` is the effect-scale
    # conditional mean u = D b, so the row contribution is w_i * u[g(i)] and the
    # exported mode is b = u ./ sd_group, the same latent scale as every other
    # family's mode (b ~ N(0, C), effect = sd * b).
    report_sd && !reml && _attach_conditional_structured!(
        scales, w .* blup[gidx], blup ./ exp.(Zg * θ̂[(pμ+pσ+1):end]))
    fit = _withranef(_withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n,
                                     converged, means, obs, scales), nll_ml), re)
    if reml
        return _withreml(fit, -nll_reml(θ̂), -nll_ml(θ̂))
    end
    return fit
end

# Kernel for coordinate `spatial(0 + x | site, coords = ...)` SD regression.
# Matches native drmTMB (`drm_spatial_coords_precision`): exponential
# correlation exp(-d / range) with range = the median positive pairwise
# distance (maximum if the median is not positive), plus 1e-6 on the diagonal.
# The range is FIXED, on the intercept-only route too (it is never estimated).
# Rows of `coords` follow the first-seen levels of the group.
function _gaussian_spatial_sd_kernel(coords, G::Int, grp::Symbol)
    # A 1-D transect may be a plain vector: read it as the G-by-1 column it means.
    cmat = coords isa AbstractVector{<:Real} ? reshape(Float64.(coords), :, 1) :
           Matrix{Float64}(coords)
    size(cmat, 1) == G ||
        throw(ArgumentError("coords must have $G rows (one per `$grp` level)"))
    all(isfinite, cmat) || throw(ArgumentError("coords must be finite"))
    Dd = [sqrt(sum(abs2, cmat[k, :] .- cmat[l, :])) for k in 1:G, l in 1:G]
    pos = [Dd[k, l] for k in 1:G for l in (k+1):G if Dd[k, l] > 0]
    isempty(pos) && throw(ArgumentError(
        "Spatial coordinates for `$grp` contain no positive distances; " *
        "at least two sites must have distinct coordinates."))
    rng = median(pos)
    (isfinite(rng) && rng > 0) || (rng = maximum(pos))
    return exp.(-Dd ./ rng) + 1e-6 * I
end

# Router leg: Gaussian structured SD regression with ONE structured coefficient,
#   y_i = X_i β + w_i u_{g(i)} + ε_i,  u = D b,  b ~ N(0, C),
#   D = diag(exp(Z α)),  log σ_i = Xσ βσ,
# where w_i = x_i for the slope-only marker `animal|relmat(0 + x | g)` and
# coordinate `spatial(0 + x | g, coords)` (`slope::Symbol`), and w_i = 1 for the
# intercept marker `animal|relmat(1 | g)` / `spatial(1 | g, coords)`
# (`slope === nothing`), each with `sd(g, animal|relmat|spatial) ~ z`.
# Scale choice (matches native drmTMB): `A` and `K` enter RAW — the diagonal is
# not normalised to 1, so exp(Z α) multiplies a latent effect whose marginal SD
# is exp(Z α) sqrt(C_gg). (Poisson/Gamma relmat in Julia normalise K; Gaussian
# does not.) The marginal likelihood is exact (dense n×n Cholesky).
function _drm_gaussian_structured_sd_slope(f::DrmFormula, fam::Gaussian, structured,
                                           slope::Union{Symbol,Nothing}, re, sigma_re, metav,
                                           structured_sigma, has_missing_response,
                                           y, Xμ, Xσ, nmμ, nmσ, data, K, A, coords,
                                           g_tol, method, algorithm)
    kind, grp = structured
    term = slope === nothing ? "$kind(1 | $grp)" : "$kind(0 + $slope | $grp)"
    what = slope === nothing ? "intercept" : "slope"
    targets = _sd_target_parts(f)
    length(targets) == 1 || throw(ArgumentError(
        "drm: Gaussian `$term` SD regression requires exactly one `sd($grp, $kind) ~ …` formula."))
    target = only(targets)
    d = target.target
    (d.level === kind && d.dpar === :mu && d.block === nothing && d.group === grp) ||
        throw(ArgumentError("drm: Gaussian `$term` takes `sd($grp, $kind) ~ …` " *
            "targeting the structured mean coefficient (got level `$(d.level)`, group `$(d.group)`)."))
    member = slope === nothing ? :intercept : slope
    d.member in (nothing, member) || throw(ArgumentError(
        "drm: Gaussian SD target `coef($(d.member))` does not match the structured $what `$member`."))
    isempty(re) && isempty(sigma_re) && metav === nothing && structured_sigma === nothing ||
        throw(ArgumentError("drm: Gaussian structured $what SD regression takes one structured " *
            "coefficient with fixed-effect `sigma` only (no ordinary or sigma random effects, `meta_V`, or structured `sigma`)."))
    has_missing_response && throw(ArgumentError(
        "drm: Gaussian structured $what SD regression requires observed responses; remove incomplete rows first."))
    reml = method === :REML
    if reml
        # REML is admitted for the structured INTERCEPT with its own `sd(g, <level>) ~ …` formula
        # and a constant residual scale: exact (the marginal is Gaussian-linear in beta_mu) and
        # the cells drmTMB 0.8.1 fits with REML = TRUE. The slope-only form and a `sigma`
        # predictor have no R reference here, so they stay refused.
        slope === nothing || throw(ArgumentError(
            "drm: Gaussian structured $what SD regression supports maximum likelihood only for " *
            "`$term`: method = :REML is validated for the structured intercept " *
            "`$kind(1 | $grp)` with `sd($grp, $kind) ~ …` only."))
        size(Xσ, 2) == 1 || throw(ArgumentError(
            "drm: Gaussian `$term` SD regression: method = :REML requires a constant residual " *
            "scale (`sigma ~ 1`); a `sigma` predictor is not validated against drmTMB there. " *
            "Use `sigma ~ 1` or method = :ML."))
        rank(Xμ) == size(Xμ, 2) || throw(ArgumentError(
            "drm: method = :REML requires a full-rank `mu` fixed-effect design."))
    end
    algorithm === :auto || throw(ArgumentError(
        "drm: `algorithm = :$algorithm` is not implemented for Gaussian structured $what SD regression."))
    for (pname, _) in f.forms
        (pname === :mu || pname === :sigma || pname === target.key) && continue
        throw(ArgumentError("drm: Gaussian structured $what SD regression supports only `mu`, `sigma` and one SD formula."))
    end
    gidx, G = _group_index(getproperty(data, grp))
    Cmat = if kind === :relmat
        (A === nothing && coords === nothing) || throw(ArgumentError(
            "drm: Gaussian relmat SD regression uses `K = …` only."))
        K === nothing && throw(ArgumentError("$term needs the covariance matrix `K = …`"))
        Matrix{Float64}(K)
    elseif kind === :animal
        (K === nothing && coords === nothing) || throw(ArgumentError(
            "drm: Gaussian animal SD regression uses `A = …` only."))
        A === nothing && throw(ArgumentError("$term needs the relatedness matrix `A = …`"))
        Matrix{Float64}(A)
    else  # :spatial
        (K === nothing && A === nothing) || throw(ArgumentError(
            "drm: Gaussian spatial SD regression uses `coords = …` only."))
        coords === nothing && throw(ArgumentError("$term needs `coords = …`"))
        _gaussian_spatial_sd_kernel(coords, G, grp)
    end
    size(Cmat) == (G, G) || throw(ArgumentError(
        "structured matrix must be $(G)×$(G) (the number of `$grp` levels)"))
    all(isfinite, Cmat) || throw(ArgumentError("structured matrix must be finite"))
    Zg, nmsd = _sd_group_design(f.response, target.rhs, data, gidx, G, grp;
                                schema_cache = f.schema_cache, schema_key = target.key)
    size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
        "drm: Gaussian structured SD target has a rank-deficient group-level design."))
    loading = slope === nothing ? nothing :
        _poisson_animal_slope_loading(data, slope, length(y); label = "Gaussian $kind")
    return _fit_structured_gaussian_lss(fam, y, Xμ, Xσ, Zg, gidx, G, Cmat, nmμ, nmσ, nmsd,
                                        grp, g_tol; block = :sd, loading, report_sd = true, reml)
end

# Independent two-field structured intercept-plus-slope with one SD formula per
# coefficient: `y ~ x + animal|relmat(1 + x | g)`, `sd(g, <level>, coef(intercept)) ~ z0`,
# `sd(g, <level>, coef(x)) ~ z1`. Model (rho fixed at 0):
#   y_i = X_i b + s0_{g(i)} f0_{g(i)} + x_i s1_{g(i)} f1_{g(i)} + e_i,
#   f0, f1 ~ iid N(0, C) (C RAW, not normalised), s_k = exp(Z_k alpha_k).
# Exact marginal: y ~ N(X b, sigma_i^2 I + sum_k (w_k w_k') o (D_k C D_k)[g, g]),
# w_0 = 1, w_1 = x. Dense n x n Cholesky, ForwardDiff, ML only, n <= 5000.
function _drm_gaussian_structured_q2_sd(f::DrmFormula, fam::Gaussian, kind::Symbol,
                                        structured, re, sigma_re, metav, structured_sigma,
                                        has_missing_response, y, Xμ, Xσ, nmμ, nmσ, data,
                                        K, A, coords, g_tol, method, algorithm;
                                        tree = nothing)
    fn = _q2_marker(kind)
    level = _q2_target_level(kind)
    block = _animal_intercept_slope(Dict(f.forms)[:mu]; kind = fn,
                                    allow_label = true)
    grp, slope = block.group, block.slope
    correlated = block.label !== nothing
    term = correlated ? "$kind(1 + $slope | $(block.label) | $grp)" : "$kind(1 + $slope | $grp)"
    structured == (kind, grp) || throw(ArgumentError(
        "drm: Gaussian `$term` SD regression requires it to be the only structured term."))
    _collect_structured(Dict(f.forms)[:mu]) == [(kind, grp)] || throw(ArgumentError(
        "drm: Gaussian `$term` SD regression requires exactly one structured term."))
    targets = _sd_target_parts(f)
    length(targets) == 2 || throw(ArgumentError(
        "drm: Gaussian `$term` needs one SD formula for the intercept and one for the slope " *
        "(`sd($grp, $level, coef(intercept)) ~ …` and `sd($grp, $level, coef($slope)) ~ …`)."))
    all(item -> item.target.level === level && item.target.group === grp &&
        item.target.dpar === :mu && item.target.block === nothing, targets) ||
        throw(ArgumentError("drm: Gaussian `$term` SD regression requires two matching `$level` mean targets."))
    isempty(_sdphylo_parts(f)) || throw(ArgumentError(
        "drm: Gaussian `$term` SD regression cannot be combined with an intercept-only " *
        "`sd($grp, phylogenetic) ~ …` formula."))
    Set(item.target.member for item in targets) == Set((:intercept, slope)) ||
        throw(ArgumentError("drm: Gaussian `$term` SD formulas must select " *
            "`coef(intercept)` and `coef($slope)` exactly once."))
    isempty(re) && isempty(sigma_re) && metav === nothing && structured_sigma === nothing ||
        throw(ArgumentError("drm: Gaussian `$term` SD regression takes fixed-effect `sigma` only " *
            "(no ordinary or sigma random effects, `meta_V`, or structured `sigma`)."))
    has_missing_response && throw(ArgumentError(
        "drm: Gaussian `$term` SD regression requires observed responses; remove incomplete rows first."))
    reml = method === :REML
    if reml
        # Native drmTMB admits REML for the structured q2 SD cells it has validated:
        # `phylo` (labelled or not, any fixed-effect `sigma`), and `animal` / `relmat` /
        # `spatial` only UNLABELLED with a constant residual scale (`sigma ~ 1`).
        # The gate mirrors that exactly (drmTMB `drm_validate_reml_spec`,
        # `drm_reml_admits_mean_structured_provider`); everything else stays refused
        # rather than being relabelled ML.
        if kind !== :phylo
            correlated && throw(ArgumentError(
                "drm: Gaussian `$term` SD regression: method = :REML is not implemented for the " *
                "labelled (correlated) `$kind` block; native drmTMB refuses it too. Use the " *
                "unlabelled `$kind(1 + $slope | $grp)` or method = :ML."))
            size(Xσ, 2) == 1 || throw(ArgumentError(
                "drm: Gaussian `$term` SD regression: method = :REML requires a constant residual " *
                "scale (`sigma ~ 1`) for `$kind`; native drmTMB refuses heteroscedastic sigma here " *
                "too. Use `sigma ~ 1` or method = :ML."))
        end
        rank(Xμ) == size(Xμ, 2) || throw(ArgumentError(
            "drm: method = :REML requires a full-rank `mu` fixed-effect design."))
    end
    algorithm === :auto || throw(ArgumentError(
        "drm: `algorithm = :$algorithm` is not implemented for Gaussian `$term` SD regression."))
    allowed = Set(vcat([:mu, :sigma], [item.key for item in targets]))
    all(first(pair) in allowed for pair in f.forms) || throw(ArgumentError(
        "drm: Gaussian `$term` SD regression supports only `mu`, `sigma` and its two SD formulas."))
    gidx, G = kind in (:phylo, :spatial) ? (Int[], 0) : _group_index(getproperty(data, grp))
    Cmat = if kind in (:phylo, :spatial)
        Ck, gidx, G = _structured_q2_kernel(kind, grp, getproperty(data, grp);
            tree, K, A, coords, who = "drm: Gaussian `$term` SD regression")
        Ck
    elseif kind === :relmat
        (A === nothing && coords === nothing && tree === nothing) || throw(ArgumentError(
            "drm: Gaussian relmat SD regression uses `K = …` only."))
        K === nothing && throw(ArgumentError("$term needs the covariance matrix `K = …`"))
        Matrix{Float64}(K)
    else
        (K === nothing && coords === nothing && tree === nothing) || throw(ArgumentError(
            "drm: Gaussian animal SD regression uses `A = …` only."))
        A === nothing && throw(ArgumentError("$term needs the relatedness matrix `A = …`"))
        Matrix{Float64}(A)
    end
    size(Cmat) == (G, G) || throw(ArgumentError(
        "structured matrix must be $(G)×$(G) (the number of `$grp` levels)"))
    all(isfinite, Cmat) || throw(ArgumentError("structured matrix must be finite"))
    ordered = map((:intercept, slope)) do member
        only(filter(item -> item.target.member === member, targets))
    end
    Zgs = Matrix{Float64}[]
    nmsds = Vector{String}[]
    for item in ordered
        Zg, names = _sd_group_design(f.response, item.rhs, data, gidx, G, grp;
                                     schema_cache = f.schema_cache, schema_key = item.key)
        size(Zg, 2) <= G && rank(Zg) == size(Zg, 2) || throw(ArgumentError(
            "drm: Gaussian `$term` SD target `coef($(item.target.member))` has a rank-deficient group-level design."))
        push!(Zgs, Matrix{Float64}(Zg)); push!(nmsds, names)
    end
    x = _poisson_animal_slope_loading(data, slope, length(y); label = "Gaussian $kind")
    if all(==(first(x)), x) && isequal(Zgs[1], Zgs[2])
        throw(ArgumentError("drm: Gaussian `$term` SD regression cannot separate the intercept and " *
            "slope variances when `$slope` is constant and both SD formulas have identical group-level designs."))
    end
    blocks = Symbol[_sd_target_block(item.target; disambiguate = true) for item in ordered]
    return _fit_structured_gaussian_q2_sd(fam, y, Xμ, Xσ, Zgs, gidx, G, Cmat, x,
                                          nmμ, nmσ, nmsds, blocks, grp, g_tol; correlated, reml)
end

# `correlated = true` (labelled `kind(1 + x | p | g)`) adds drmTMB's
# `eta_cor_phylo` as the last parameter: rho = 0.999999 tanh(eta) and
# Cov(b0_g, b1_h) = rho s0_g s1_h C[g, h], so the marginal gains
#   rho (s0_g s1_h x_j + x_i s1_g s0_h) C[g(i), g(j)].
function _fit_structured_gaussian_q2_sd(fam::Gaussian, y, Xμ, Xσ, Zgs, gidx, G, C, x,
                                        nmμ, nmσ, nmsds, blocks, grp, g_tol;
                                        correlated::Bool = false, reml::Bool = false)
    n = length(y)
    n ≤ 5000 || throw(ArgumentError("drm: the Gaussian structured intercept-and-slope SD route " *
        "assembles a $(n)×$(n) marginal covariance and is limited to 5000 rows."))
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    ps = size.(Zgs, 2)
    r0 = (pμ + pσ + 1):(pμ + pσ + ps[1])
    r1 = (pμ + pσ + ps[1] + 1):(pμ + pσ + ps[1] + ps[2])
    Csym = Symmetric(_finite_user_matrix(C, "C"))
    const_2pi = 0.5 * n * log(2π)
    const_pμ = 0.5 * pμ * log(2π)
    # One objective for ML and REML. REML (Patterson-Thompson, the restriction native
    # drmTMB gets by moving `beta_mu` into the Laplace block, exact here because the
    # likelihood is quadratic in beta_mu) adds `0.5 logdet(Xμ' V^-1 Xμ) - 0.5 pμ log 2π`
    # to the ML negative log-likelihood; the optimiser still carries βμ in θ, and at the
    # optimum it equals the GLS estimate because the added term does not depend on βμ.
    # Marginal covariance V(θ) = diag(σ²) + the structured field(s) (β_μ plays no part).
    function build_V(θ)
        βσ = θ[pμ+1:pμ+pσ]
        ησ = Xσ * βσ
        T = eltype(θ)
        s0 = exp.(Zgs[1] * θ[r0]); s1 = exp.(Zgs[2] * θ[r1])
        Σ0 = (s0 * s0') .* Csym; Σ1 = (s1 * s1') .* Csym
        Vm = Matrix{T}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            gi, gj = gidx[i], gidx[j]
            Vm[i, j] = Σ0[gi, gj] + x[i] * x[j] * Σ1[gi, gj]
        end
        if correlated
            ρ = _Q2_RHO_BOUND * tanh(θ[end])
            Σ01 = (s0 * s1') .* Csym          # Cov(b0_g, b1_h) / ρ
            @inbounds for j in 1:n, i in 1:n
                gi, gj = gidx[i], gidx[j]
                Vm[i, j] += ρ * (x[j] * Σ01[gi, gj] + x[i] * Σ01[gj, gi])
            end
        end
        @inbounds for i in 1:n
            Vm[i, i] += exp(2 * ησ[i])
        end
        return Vm
    end
    # ML (`restricted = false`) or REML negative log-likelihood. REML (Patterson-Thompson, the
    # restriction native drmTMB gets by moving `beta_mu` into the Laplace block, exact here
    # because the likelihood is quadratic in beta_mu) adds `0.5 logdet(Xμ' V^-1 Xμ) -
    # 0.5 pμ log 2π` to the ML value. With `profile = true` the mean is the GLS estimate at
    # θ (the first pμ entries of θ are ignored): the restricted likelihood as a function of
    # the variance parameters alone, which is what TMB optimises once beta_mu is integrated.
    function nll_full(θ, restricted::Bool; profile::Bool = false)
        T = eltype(θ)
        Vfac = cholesky(Symmetric(build_V(θ)); check = false)
        issuccess(Vfac) || return convert(T, 1e18)
        XtVinvX = Xμ' * (Vfac \ Xμ)
        chX = cholesky(Symmetric(XtVinvX); check = false)
        βμ = profile ? (chX \ (Xμ' * (Vfac \ y))) : θ[1:pμ]
        r = y .- Xμ * βμ
        nll_ml_θ = 0.5 * (logdet(Vfac) + dot(r, Vfac \ r)) + const_2pi
        restricted || return nll_ml_θ
        issuccess(chX) || return nll_ml_θ + convert(T, REML_NONPD_PENALTY)
        return nll_ml_θ + 0.5 * logdet(chX) - const_pμ
    end
    nll_ml(θ) = nll_full(θ, false)
    nll_reml(θ) = nll_full(θ, true)
    nll = reml ? nll_reml : nll_ml
    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + sum(ps) + (correlated ? 1 : 0))
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) / sqrt(2) + eps())
    s_int = std(res0) / sqrt(2)
    s_slp = s_int / max(sqrt(sum(abs2, x) / n), eps())
    θ0[r0] .= Zgs[1] \ fill(log(s_int + eps()), G)
    θ0[r1] .= Zgs[2] \ fill(log(s_slp + eps()), G)
    opts = Optim.Options(g_tol = g_tol, x_abstol = NaN, x_reltol = NaN,
                         f_abstol = NaN, f_reltol = NaN)
    res = if reml
        # Optimise the restricted likelihood over the variance parameters only, with β_μ
        # profiled out by GLS (as TMB does after integrating it). Carrying β_μ as a free
        # coordinate made the fit stop in a poorer basin than native drmTMB's on a relmat
        # fixture (5 log-lik units short); the profile has no such coordinate.
        nll_prof(φ) = nll_full(vcat(zeros(eltype(φ), pμ), φ), true; profile = true)
        rφ = Optim.optimize(nll_prof, θ0[(pμ+1):end], Optim.LBFGS(), opts; autodiff = :forward)
        φ̂ = Optim.minimizer(rφ)
        Vfaĉ = _pd_chol_or_throw(build_V(vcat(zeros(pμ), φ̂)), "the REML-optimum covariance V (GLS step)")
        β̂ = Symmetric(Xμ' * (Vfaĉ \ Xμ)) \ (Xμ' * (Vfaĉ \ y))
        # Same return shape as the unrestricted route: the full vector, finished by a
        # zero-step LBFGS (the gradient in β_μ vanishes at the GLS value).
        Optim.optimize(nll, vcat(β̂, φ̂), Optim.LBFGS(), opts; autodiff = :forward)
    else
        # Same unguarded-probe hazard as the ordinary q2/q3/q4 route (see
        # `_fit_gaussian_ordinary_q34_sd`): the first L-BFGS step is the raw gradient, so a
        # probe can overflow `exp(2 eta_sigma)` or the SD scales, the objective or its AD
        # gradient is NaN, and HagerZhang asserts. Sentinel + BackTracking retry, optimiser
        # only; the reported value, Hessian and gradient check use the raw `nll`.
        _optimize_with_fallback(nll, θ0, g_tol; opts = opts)
    end
    θ̂ = Optim.minimizer(res)
    H = ForwardDiff.hessian(nll, θ̂)
    V = _vcov_from_hessian(H)
    converged = !Optim.iteration_limit_reached(res) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll, θ̂))
    block_list = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ),
                                             blocks[1] => r0, blocks[2] => r1]
    name_list = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ,
                                            blocks[1] => nmsds[1], blocks[2] => nmsds[2]]
    if correlated
        push!(block_list, :recov => length(θ̂):length(θ̂))
        push!(name_list, :recov => ["$(grp):atanh_rho"])
    end
    sd0 = exp.(Zgs[1] * θ̂[r0]); sd1 = exp.(Zgs[2] * θ̂[r1])
    scales = Dict{Symbol,Vector{Float64}}(
        :sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]),
        blocks[1] => sd0[gidx], blocks[2] => sd1[gidx],
        Symbol(string(blocks[1]), "_group") => sd0,
        Symbol(string(blocks[2]), "_group") => sd1)
    # Conditional modes at θ̂ (closed form: the model is Gaussian). With V = Σ_re + D,
    # D = diag(σ²), E[random part | y] = Σ_re V⁻¹ r = r − D V⁻¹ r, and the unit-scale
    # field modes are E[z_k | y] = Cov(z_k, y) V⁻¹ r (z0, z1 ~ N(0, C), Cov(z0, z1) = ρ C).
    # `build_V` already carries the residual diagonal D.
    let Vfull = build_V(θ̂), σ2 = exp.(2 .* (Xσ * θ̂[(pμ+1):(pμ+pσ)])),
        r = y .- Xμ * θ̂[1:pμ]
        Vfac = cholesky(Symmetric(Vfull); check = false)
        if issuccess(Vfac)
            w = Vfac \ r
            eta_re = r .- σ2 .* w
            ρ = correlated ? _Q2_RHO_BOUND * tanh(θ̂[end]) : 0.0
            Cm = Matrix(Csym)
            s0, s1 = sd0[gidx], sd1[gidx]
            Cz0 = Cm[:, gidx] .* (s0 .+ ρ .* x .* s1)'     # Cov(z0_g, y_i)
            Cz1 = Cm[:, gidx] .* (x .* s1 .+ ρ .* s0)'     # Cov(z1_g, y_i)
            _attach_conditional_structured!(scales, eta_re, Cz0 * w, Cz1 * w)
        else
            _mark_cond_failed!(scales)
        end
    end
    fit = DrmFit(fam, block_list, name_list, θ̂, V, -nll(θ̂), n, converged,
                 Dict(:mu => Xμ * θ̂[1:pμ]), Dict(:mu => Vector{Float64}(y)), scales)
    fit = _withiterations(_withnll(fit, nll), Optim.iterations(res))
    reml && return _withreml(fit, -nll_reml(θ̂), -nll_ml(θ̂))
    return fit
end

# ---------------------------------------------------------------------------
# #555 — location-scale-scale-SCALE: several random effects, each with its own
# SD submodel, in one fit. drmTMB accepts e.g.
#     bf(y ~ x + (1|species) + (1|study) + phylo(1|species, tree = tr),
#        sigma ~ 1, sd(species) ~ x, sd(study) ~ 1,
#        sd(species, level = "phylogenetic") ~ x)
# (verified by direct fits, 2026-08-28). Marginal:
#     V = Σ_k Z_k D_k² Z_k'  +  Z_p (D_a K D_a) Z_p'  +  diag(σ_e,i²)
# with each D a per-group diagonal exp(Zg α) from its own linear predictor
# (a scalar SD is the `~ 1` special case). Dense assembly, ML, ForwardDiff —
# same correctness-first stance and 5000-row cap as the dense single-component route; the iid
# α's share one `:sd` block (names group-prefixed when there is more than one
# iid component) and the phylo α's are the `:sd_phylo` block.

# One variance component of the multi engine.
# Marker for "this component IS phylogenetic, but its dense G x G tree
# correlation has not been materialised yet" (#563 S7b.6). The router builds
# every phylo `_LssComp` with this marker instead of eagerly calling the
# CUBIC `_phylo_correlation` — the sparse route never reads `.K` as a matrix
# (it rebuilds the sparse precision from `phy_tree` directly), so computing
# the dense inverse before the router even decides `use_sparse` wasted O(p^3)
# work on every sparse-route fit. Only the dense route materialises it.
struct _LazyPhyloK end

struct _LssComp
    gidx::Vector{Int}
    G::Int
    Zg::Matrix{Float64}
    nm::Vector{String}
    K::Union{Nothing,Matrix{Float64},_LazyPhyloK}   # nothing = iid; matrix = phylo correlation; _LazyPhyloK = phylo, not yet materialised
    label::String
    w::Union{Nothing,Vector{Float64}}                # per-row loading of an iid component (an independent slope `(0 + x | g)`); nothing = intercept
end
_LssComp(gidx, G, Zg, nm, K, label) = _LssComp(gidx, G, Zg, nm, K, label, nothing)

function _fit_gaussian_lss_multi(fam::Gaussian, y, Xμ, Xσ, comps::Vector{_LssComp},
                                 nmμ, nmσ, g_tol; reml::Bool = false)
    n = length(y)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    const_2pi = 0.5 * n * log(2π)
    const_pμ = 0.5 * pμ * log(2π)
    n ≤ 5000 || throw(ArgumentError("drm: the multi-component sd() route assembles a dense " *
        "$(n)×$(n) marginal covariance and is limited to 5000 rows in this slice (#555)."))
    psds = [size(c.Zg, 2) for c in comps]
    offs = pμ + pσ .+ cumsum([0; psds])           # α_c = θ[offs[c]+1 : offs[c+1]]

    function nll_ml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        T = eltype(θ)
        Vm = zeros(T, n, n)
        for (ci, c) in enumerate(comps)
            α = θ[offs[ci]+1:offs[ci+1]]
            σa = exp.(c.Zg * α)
            if c.K === nothing
                # iid: Z Diagonal(σa²) Z' adds σa[g]² wherever rows share a group
                @inbounds for j in 1:n, i in 1:n
                    if c.gidx[i] == c.gidx[j]
                        Vm[i, j] += σa[c.gidx[i]]^2 * (c.w === nothing ? 1.0 : c.w[i] * c.w[j])
                    end
                end
            else
                Σc = (σa * σa') .* c.K
                @inbounds for j in 1:n, i in 1:n
                    Vm[i, j] += Σc[c.gidx[i], c.gidx[j]]
                end
            end
        end
        @inbounds for i in 1:n
            Vm[i, i] += exp(2 * ησ[i])
        end
        Vfac = cholesky(Symmetric(Vm); check = false)
        issuccess(Vfac) || return convert(T, 1e18)
        r = y .- ημ
        return 0.5 * (logdet(Vfac) + dot(r, Vfac \ r)) + const_2pi
    end

    function nll_reml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        T = eltype(θ)
        Vm = zeros(T, n, n)
        for (ci, c) in enumerate(comps)
            α = θ[offs[ci]+1:offs[ci+1]]
            σa = exp.(c.Zg * α)
            if c.K === nothing
                @inbounds for j in 1:n, i in 1:n
                    if c.gidx[i] == c.gidx[j]
                        Vm[i, j] += σa[c.gidx[i]]^2 * (c.w === nothing ? 1.0 : c.w[i] * c.w[j])
                    end
                end
            else
                Σc = (σa * σa') .* c.K
                @inbounds for j in 1:n, i in 1:n
                    Vm[i, j] += Σc[c.gidx[i], c.gidx[j]]
                end
            end
        end
        @inbounds for i in 1:n
            Vm[i, i] += exp(2 * ησ[i])
        end
        Vfac = cholesky(Symmetric(Vm); check = false)
        issuccess(Vfac) || return convert(T, 1e18)
        r = y .- ημ
        nll_ml_θ = 0.5 * (logdet(Vfac) + dot(r, Vfac \ r)) + const_2pi
        A = Vfac \ Xμ
        XtVinvX = Xμ' * A
        chX = cholesky(Symmetric(XtVinvX); check = false)
        issuccess(chX) || return nll_ml_θ + convert(T, REML_NONPD_PENALTY)
        return nll_ml_θ + 0.5 * logdet(chX) - const_pμ
    end

    nll = reml ? nll_reml : nll_ml

    βμ0 = Xμ \ y; res0 = y - Xμ * βμ0
    m = length(comps)
    θ0 = zeros(offs[end])
    θ0[1:pμ] .= βμ0
    θ0[pμ+1] = log(std(res0) / sqrt(m + 1) + eps())
    for (ci, c) in enumerate(comps)
        θ0[offs[ci]+1:offs[ci+1]] .= c.Zg \ fill(log(std(res0) / sqrt(m + 1) + eps()), c.G)
    end
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    H = ForwardDiff.hessian(nll, θ̂)
    Vcov = _vcov_from_hessian(H)
    # Same flag as the scalar route: not the iteration cap, and the Newton step
    # in SE units. `g_converged` and the x/f/g OR stay out (#983). A non-finite
    # Hessian is not converged (#1114).
    converged = !Optim.iteration_limit_reached(res) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll, θ̂))

    iid = [ci for (ci, c) in enumerate(comps) if c.K === nothing]
    phy = [ci for (ci, c) in enumerate(comps) if c.K !== nothing]
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ)]
    names = Pair{Symbol,Vector{String}}[:mu => nmμ, :sigma => nmσ]
    # the iid α's are one contiguous :sd block ONLY when the components are
    # laid out contiguously — the builder below guarantees iid-then-phylo order.
    if !isempty(iid)
        lo = offs[first(iid)] + 1; hi = offs[last(iid)+1]
        nm = String[]
        for ci in iid
            c = comps[ci]
            append!(nm, length(iid) > 1 ? string.(c.label, ": ", c.nm) : c.nm)
        end
        push!(blocks, :sd => lo:hi); push!(names, :sd => nm)
    end
    if !isempty(phy)
        ci = only(phy)
        push!(blocks, :sd_phylo => (offs[ci]+1):offs[ci+1])
        push!(names, :sd_phylo => comps[ci].nm)
    end
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    fit = _withnll(DrmFit(fam, blocks, names, θ̂, Vcov, -nll(θ̂), n,
                          converged, means, obs, scales), nll_ml)
    if reml
        return _withreml(fit, -nll_reml(θ̂), -nll_ml(θ̂))
    end
    return fit
end

# ---------------------------------------------------------------------------
# #563 S7b.4 — router rule (D-206, design note §1/§7): dispatch a multi-
# component sd() model to the sparse route ONLY when exactly one component is
# phylogenetic and every iid component is either NESTED within it (each iid
# group maps to a single phylo group) or SMALL (`G_c ≤ _LSS_SPARSE_SMALL_FRAC
# × G_phy`, documented here as the threshold rule) — otherwise the dense
# multi-component route, even under `algorithm = :sparse`. This mirrors the
# GO (one phylo + nested/small iid) / NO-GO (comparable-size crossed factors)
# classification the design note's fill measurements (§2.1/§2.3) establish.

# "Small enough to route sparse even though its grouping is not verified
# nested" — `G_c ≪ p` in the design note's own words (§1, §7's `S7b.2`
# estimate row); 10% of the phylogenetic component's tip count is the
# documented rule for this sub-slice.
const _LSS_SPARSE_SMALL_FRACTION = 0.1

# Is `gidx_child` a NESTED grouping within `gidx_parent`, i.e. does every
# child group level map to exactly one parent group level across all rows?
# (A clean partition — no observation's child-group membership is
# independent of its parent-group membership; design note §1/§2.)
function _lss_iid_nested_in(gidx_child::Vector{Int}, gidx_parent::Vector{Int})
    parent_of = Dict{Int,Int}()
    for i in eachindex(gidx_child)
        g, p = gidx_child[i], gidx_parent[i]
        if haskey(parent_of, g)
            parent_of[g] == p || return false
        else
            parent_of[g] = p
        end
    end
    return true
end

# Eligibility for the sparse multi-component route: exactly one phylogenetic
# component, and every iid component nested-in-or-small-relative-to it.
# Returns `(eligible::Bool, reason::String)`; `reason` is empty when eligible.
function _lss_multi_sparse_eligible(comps::Vector{_LssComp})
    phy_positions = findall(c -> c.K !== nothing, comps)
    if length(phy_positions) != 1
        reason = isempty(phy_positions) ? "no phylogenetic sd() component" :
                                           "more than one phylogenetic sd() component"
        return false, reason
    end
    pi = only(phy_positions)
    phy = comps[pi]
    for (ci, c) in enumerate(comps)
        ci == pi && continue
        nested = _lss_iid_nested_in(c.gidx, phy.gidx)
        small = c.G <= _LSS_SPARSE_SMALL_FRACTION * phy.G
        if !nested && !small
            reason = "component `$(c.label)` ($(c.G) levels) is CROSSED with " *
                "`$(phy.label)` ($(phy.G) tips) and not small enough " *
                "(needs ≤ $(_LSS_SPARSE_SMALL_FRACTION) × $(phy.G) = " *
                "$(_LSS_SPARSE_SMALL_FRACTION * phy.G) levels) — the sparse " *
                "route's O(p) fill guarantee (design note §2.1/§2.3) does not " *
                "apply to this topology"
            return false, reason
        end
    end
    return true, ""
end

# Build the sparse `_SparseLssComp` list (iid-then-phylo order, matching the
# dense route's contiguous-block layout convention, `gaussian_lss.jl:659-
# 669`) plus its parallel label/name/is-phylo metadata, from the SAME
# (possibly missing-response-filtered) `comps::Vector{_LssComp}` the dense
# route uses and the phylogenetic component's tree.
function _lss_multi_sparse_comps(comps::Vector{_LssComp}, phy_tree)
    phy_positions = findall(c -> c.K !== nothing, comps)
    pi = only(phy_positions)
    order = vcat([ci for ci in eachindex(comps) if ci != pi], pi)
    sp_comps = _SparseLssComp[]
    comp_nm = Vector{String}[]
    comp_is_phylo = Bool[]
    comp_label = String[]
    for ci in order
        c = comps[ci]
        if ci == pi
            push!(sp_comps, _sparse_lss_phylo_comp(c.gidx, c.G, c.Zg, phy_tree))
            push!(comp_is_phylo, true)
        else
            push!(sp_comps, _sparse_lss_iid_comp(c.gidx, c.G, c.Zg))
            push!(comp_is_phylo, false)
        end
        push!(comp_nm, c.nm)
        push!(comp_label, c.label)
    end
    return sp_comps, comp_nm, comp_is_phylo, comp_label
end

# Build the component list from parsed formula parts and dispatch. Handles every
# sd() request the single-component routes do not: several iid REs, iid + phylo,
# an RE without an sd() part (scalar, `~ 1`), and a phylo term without its own
# sd() part alongside sd()-carrying iid REs.
#
# `algorithm`/`sparse` (#563 S7b.4): forwarded from `drm(...)` exactly as the
# single-component `_drm_gaussian_lss_phylo` route already accepts them
# (`gaussian_core.jl` — the sibling call two lines above this route's own).
# `:auto` (default) auto-dispatches to sparse when eligible (above) AND the
# phylogenetic component has more than 500 tips, mirroring the single-
# component `G > 500` rule (`:402` above). `:sparse`/`:sparse_lbfgs`/`sparse =
# true` FORCE the sparse route when eligible and fall back to dense — with an
# informative `@info` — when the model is not (D-206's NO-GO for genuinely
# crossed topologies): never silently drop the request, never silently run an
# unproven-fill sparse fit either.
function _drm_gaussian_lss_multi(f::DrmFormula, fam::Gaussian, sdp, sdpp, re, re_kinds,
                                 structured, has_missing_response, y, Xμ, Xσ, nmμ, nmσ, data, tree, g_tol, method;
                                 algorithm::Symbol = :auto, sparse = nothing)
    for (kind, _) in re_kinds
        kind === :intercept ||
            throw(ArgumentError("drm: sd() with several random-effect terms supports random intercepts " *
                "`(1 | g)` here. Independent slopes `(0 + x | g)` take `sd(g, coef(x)) ~ …` on ordinary " *
                "groupings, and one correlated block `(1 + x | g)` beside random intercepts on other " *
                "groupings is supported; this combination is not implemented."))
    end
    re_groups = Symbol[grp for (_, grp) in re]
    for (sgrp, _) in sdp
        sgrp in re_groups ||
            throw(ArgumentError("drm: `sd($sgrp)` must name the grouping of a `(1 | $sgrp)` " *
                "random effect in the mean formula (random effects present: " *
                join(re_groups, ", ") * ")."))
    end
    comps = _LssComp[]
    phy_tree = nothing   # captured for the sparse router branch below (#563 S7b.4)
    sdmap = Dict(sdp)
    for grp in re_groups                          # iid components first (block layout)
        gidx, G = _group_index(getproperty(data, grp); col = grp)
        if haskey(sdmap, grp)
            Zg, nm = _sd_group_design(f.response, sdmap[grp], data, gidx, G, grp)
        else
            Zg, nm = ones(G, 1), ["(Intercept)"]  # RE without sd() part: scalar SD
        end
        push!(comps, _LssComp(gidx, G, Zg, nm, nothing, String(grp)))
    end
    if structured !== nothing
        kind, pgrp = structured
        kind === :phylo ||
            throw(ArgumentError("drm: sd() submodels with `$kind(1 | $pgrp)` are not " *
                "supported — the structured scale level is `phylogenetic` only (#555)."))
        tree === nothing && error("phylo(1 | $pgrp) needs `tree = …`")
        phy, gidx, G = _lss_phylo_group_index(tree, getproperty(data, pgrp), pgrp)
        phy_tree = phy
        # #563 S7b.6: do NOT materialise the dense G×G tree correlation here —
        # the router below may pick the sparse route, which never reads it as
        # a matrix (it rebuilds a sparse precision from `phy_tree` directly).
        # `_LazyPhyloK()` marks this component as phylogenetic; only the dense
        # route (below) calls `_phylo_correlation` and checks its size.
        if !isempty(sdpp)
            (sgrp, srhs) = only(sdpp)
            sgrp === pgrp ||
                throw(ArgumentError("drm: `sd($sgrp, phylogenetic)` must name the phylo " *
                    "grouping — the mean formula has `phylo(1 | $pgrp)`."))
            Zg, nm = _sd_group_design(f.response, srhs, data, gidx, G, pgrp)
        else
            Zg, nm = ones(G, 1), ["(Intercept)"]  # phylo term with scalar SD
        end
        push!(comps, _LssComp(gidx, G, Zg, nm, _LazyPhyloK(), String(pgrp)))
    elseif !isempty(sdpp)
        (sgrp, _) = only(sdpp)
        throw(ArgumentError("drm: `sd($sgrp, phylogenetic)` requires a `phylo(1 | $sgrp)` " *
            "random effect in the mean formula."))
    end
    isempty(comps) &&
        throw(ArgumentError("drm: sd() formulas need matching random effects in the mean formula."))
    if has_missing_response
        observed = _observed_response_mask(y)
        n_obs = count(observed)
        pμ, pσ = size(Xμ, 2), size(Xσ, 2)
        psds = [size(c.Zg, 2) for c in comps]
        total_dof = pμ + pσ + sum(psds)
        n_obs >= total_dof ||
            throw(ArgumentError("drm (Gaussian sd multi): only $(n_obs) observed responses for a model with " *
                "$(total_dof) parameters ($(pμ) mean + $(pσ) scale + $(sum(psds)) sd) — " *
                "too few to fit. Use `drm_listwise` or supply more complete responses."))
        n_obs > total_dof ||
            @warn "drm (Gaussian sd multi): $(n_obs) observed responses equals the $(total_dof) model " *
                  "parameters (residual dof 0); the fit is saturated and inference is unreliable."
        @warn "drm: $(length(observed) - n_obs) of $(length(observed)) rows have a missing/NaN " *
              "response and were dropped (observed-rows fit, like glmmTMB's default na.action). " *
              "Use `drm_listwise` to preprocess explicitly, or supply complete responses, to silence this."
        y = Vector{Float64}(y[observed])
        Xμ = Matrix{Float64}(Xμ[observed, :])
        Xσ = Matrix{Float64}(Xσ[observed, :])
        comps = [_LssComp(c.gidx[observed], c.G, c.Zg, c.nm, c.K, c.label) for c in comps]
    end

    # #563 S7b.4 router (D-206): decide sparse vs. dense for this model.
    eligible, ineligible_reason = _lss_multi_sparse_eligible(comps)
    sparse_requested = (sparse === true) || (algorithm in (:sparse, :sparse_lbfgs))
    phy_positions = findall(c -> c.K !== nothing, comps)
    use_sparse = if sparse_requested
        eligible
    elseif algorithm === :auto
        eligible && !isempty(phy_positions) && comps[only(phy_positions)].G > 500
    else
        false   # :gls / :lbfgs / :em have no sparse-multi meaning; stay dense
    end
    if sparse_requested && !eligible
        @info "drm: sd() multi-component model requested `algorithm = :$(algorithm === :auto ? :sparse : algorithm)`" *
              "$(sparse === true ? " (sparse = true)" : "") but is not eligible for the sparse " *
              "multi-component route ($ineligible_reason); using the dense multi-component " *
              "route instead."
    end

    if use_sparse
        sp_comps, comp_nm, comp_is_phylo, comp_label = _lss_multi_sparse_comps(comps, phy_tree)
        fit = _fit_gaussian_lss_sparse_multi(fam, y, Xμ, Xσ, sp_comps, comp_nm, comp_is_phylo,
                                             comp_label, nmμ, nmσ, g_tol; reml = method === :REML)
        return _mark_lss_multi_route!(fit, :sparse_multi)
    end
    # Dense route: materialise the phylo component's dense G×G correlation
    # NOW (#563 S7b.6) — the sparse branch above never reaches this point, so
    # the cubic `_phylo_correlation` cost is paid only when it is actually
    # needed.
    dense_comps = [c.K isa _LazyPhyloK ? _materialize_lss_comp_K(c, phy_tree) : c for c in comps]
    fit = _fit_gaussian_lss_multi(fam, y, Xμ, Xσ, dense_comps, nmμ, nmσ, g_tol; reml = method === :REML)
    return _mark_lss_multi_route!(fit, :dense_multi)
end

# Replace a `_LazyPhyloK()` marker with the materialised dense G×G tree
# correlation (#563 S7b.6) — same size check the eager computation had.
function _materialize_lss_comp_K(c::_LssComp, phy_tree)
    Kmat = _phylo_correlation(phy_tree)
    size(Kmat) == (c.G, c.G) ||
        error("structured matrix must be $(c.G)×$(c.G) (the number of `$(c.label)` levels)")
    return _LssComp(c.gidx, c.G, c.Zg, c.nm, Matrix{Float64}(Kmat), c.label)
end

function _drm_gaussian_lss_multi(f::DrmFormula, fam::Gaussian, sdp, sdpp, re, re_kinds,
                                 structured, y, Xμ, Xσ, nmμ, nmσ, data, tree, g_tol, method;
                                 algorithm::Symbol = :auto, sparse = nothing)
    return _drm_gaussian_lss_multi(f, fam, sdp, sdpp, re, re_kinds, structured, false,
                                   y, Xμ, Xσ, nmμ, nmσ, data, tree, g_tol, method;
                                   algorithm = algorithm, sparse = sparse)
end
