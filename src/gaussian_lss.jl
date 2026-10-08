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
bf(y ~ x + (1 | g),              sigma ~ x, sd(g) ~ z)                  # iid RE SD
bf(y ~ x + phylo(1 | species),   sigma ~ x, sd(species, phylogenetic) ~ z)  # phylo SD
```

to put a linear predictor on the **log standard deviation of a random
effect**. Plain `sd(g)` targets the iid `(1 | g)` intercept
(`log σ_b,k = Z_k' α`, one row per group level); `sd(group, phylogenetic)`
targets the per-species phylogenetic SD (drmTMB:
`sd(group, level = "phylogenetic")`; `@formula` cannot parse keyword
arguments, so the level is a bare symbol). Predictors must be constant
within each group level. Univariate Gaussian only; coefficients appear as
the `:sd` (iid) or `:sd_phylo` (phylogenetic) block.

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

# Extract `sd(g) ~ rhs` parts stashed by `bf` as `Symbol("sd_", g) => rhs`.
_sd_parts(f::DrmFormula) = Pair{Symbol,Any}[
    Symbol(String(first(p))[(length("sd_")+1):end]) => last(p)
    for p in f.forms if startswith(String(first(p)), "sd_")]

# One-line guard for family routes that do not support sd(): every family
# consumer reads `Dict(f.forms)` by known keys, so an unguarded sd() part would
# be DROPPED SILENTLY — the exact bug class of issue #2 (silent σ-phylo drop).
function _lss_only_gaussian_guard(f::DrmFormula, fam)
    (isempty(_sd_parts(f)) && isempty(_sdphylo_parts(f))) ||
        throw(ArgumentError("drm: `sd(group) ~ …` location-scale-scale formulas are " *
            "supported for the univariate Gaussian family only (got $(nameof(typeof(fam))))."))
    return nothing
end

# Group-level design for the sd() submodel: build the n-row design from the full
# data, require every predictor constant within each group level (drmTMB errors
# the same way), and compress to one row per group in the supplied `gidx` order.
# IID callers use `_group_index`'s first-seen order; phylogenetic callers supply
# the tree-tip order through `_lss_phylo_group_index` below.
function _sd_group_design(response::Symbol, rhs, data, gidx::Vector{Int}, G::Int, grp::Symbol)
    _, Zfull, nm = _design(response, rhs, data)
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
    (structured === nothing && structured_sigma === nothing) ||
        throw(ArgumentError("drm: `sd($sgrp)` with structured effects (phylo/relmat/animal/" *
            "spatial) is not yet supported — the climate-dependent phylogenetic SD " *
            "(`sd_phylo`) is issue #545."))
    metav === nothing ||
        throw(ArgumentError("drm: `sd($sgrp)` cannot be combined with `meta_V(...)`."))
    isempty(sigma_re) ||
        throw(ArgumentError("drm: `sd($sgrp)` cannot be combined with a random effect on `sigma`."))
    length(re) == 1 ||
        throw(ArgumentError("drm: `sd($sgrp)` requires exactly one `(1 | $sgrp)` random " *
            "intercept on the mean (got $(length(re)) random-effect terms)."))
    (kind, _) = re_kinds[1]
    kind === :intercept ||
        throw(ArgumentError("drm: `sd($sgrp)` supports a random INTERCEPT `(1 | $sgrp)` only."))
    (_, grp) = re[1]
    grp === sgrp ||
        throw(ArgumentError("drm: `sd($sgrp)` must name the grouping of the random effect — " *
            "the mean formula has `(1 | $grp)`."))
    gidx, G = _group_index(getproperty(data, grp))
    Zg, nmsd = _sd_group_design(f.response, srhs, data, gidx, G, grp)
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
    end
    return _fit_ranef_gaussian_lss(fam, y, Xμ, Xσ, Zg, gidx, G, nmμ, nmσ, nmsd, grp, g_tol;
                                   reml = method === :REML)
end

# Gaussian location–scale–scale with one mean random intercept (1 | g) whose
# per-group SD carries a linear predictor: θ = [β_μ; β_σ (log σ); α (log σ_b)].
# Same exact Woodbury/Patterson–Thompson marginal as `_fit_ranef_gaussian`
# (which is left byte-for-byte untouched), with σ_b² per group. Convergence
# reasoning (exact marginal, no inner-solve noise floor) carries over from that
# function's header note.
function _fit_ranef_gaussian_lss(fam::Gaussian, y, Xμ, Xσ, Zg, gidx, G, nmμ, nmσ, nmsd,
                                 grp, g_tol; reml::Bool = false)
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
            S[k] += invD
            C[k] += r * invD
            logdetD += 2 * ησ[i]
        end
        logdetCap = zero(T)
        @inbounds for k in 1:G
            σb² = exp(2 * ησb[k])
            logdetCap += log(1 + σb² * S[k])
        end
        # Cancellation-free r′V⁻¹r (#746/#747; see `_re_quad_stable`).
        quad = _re_quad_stable(rv, invDv, nothing, gidx, exp.(-2 .* ησb), S, C)
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
            S[gidx[i]] += invD
        end
        # PSD penalised-SS form of Xμ′V⁻¹Xμ. The Woodbury subtraction it replaces
        # returned exactly-singular garbage (8e110 vs a true 1e-7) at an LBFGS
        # probe with log σ_i ≈ -105; see `_re_xtvinvx_stable` and
        # test/test_lss_reml_falseconv.jl.
        XtVinvX = _re_xtvinvx_stable(Xμ, invDv, nothing, gidx, exp.(-2 .* ησb), S)
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
    # `Optim.converged` is the x/f/g OR; `g_converged` (absolute 1e-8) is not
    # part of this flag. Stopping on the iteration cap is not a minimum.
    converged = Optim.converged(res) && !Optim.iteration_limit_reached(res) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), :sd => (pμ+pσ+1):(pμ+pσ+psd)]
    names = [:mu => nmμ, :sigma => nmσ, :sd => nmsd]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]; α = θ̂[pμ+pσ+1:end]
        ημ = Xμ * βμ; ησ = Xσ * βσ; ησb = Zg * α
        S = zeros(G); C = zeros(G)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i]); k = gidx[i]
            S[k] += invD
            C[k] += (y[i] - ημ[i]) * invD
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
        throw(ArgumentError("drm: phylo LSS full input for `$grp` must contain all tree tips; absent: $absent"))
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
                                      block::Symbol = :sd_phylo, reml::Bool = false)
    n = length(y)
    pμ, pσ, psd = size(Xμ, 2), size(Xσ, 2), size(Zg, 2)
    const_2pi = 0.5 * n * log(2π)
    const_pμ = 0.5 * pμ * log(2π)
    n ≤ 5000 || throw(ArgumentError("drm: this forced dense `sd_phylo` route assembles a $(n)×$(n) " *
        "marginal covariance and is limited to 5000 rows. Use `sparse = true` or " *
        "`algorithm = :sparse_lbfgs` for the existing O(p) whole-tree engine."))
    Ksym = Symmetric(Matrix{Float64}(K))

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
            Vm[i, j] = Σa[gidx[i], gidx[j]]               # Z Σ_a Z'
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
            Vm[i, j] = Σa[gidx[i], gidx[j]]
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
    θ0[pμ+pσ+1:end] .= Zg \ fill(log(std(res0) / sqrt(2) + eps()), G)
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
    # Same flag as the scalar route: x/f/g OR, not the iteration cap, and the
    # Newton step in SE units. `g_converged` stays out (absolute 1e-8).
    converged = Optim.converged(res) && !Optim.iteration_limit_reached(res) &&
        _unitfree_converged(H, ForwardDiff.gradient(nll, θ̂))

    blocks = [:mu => 1:pμ, :sigma => (pμ+1):(pμ+pσ), block => (pμ+pσ+1):(pμ+pσ+psd)]
    names = [:mu => nmμ, :sigma => nmσ, block => nmsd]
    means = Dict(:mu => Xμ * θ̂[1:pμ])
    obs = Dict(:mu => Vector{Float64}(y))
    scales = Dict(:sigma => exp.(Xσ * θ̂[(pμ+1):(pμ+pσ)]))
    blup = let
        βμ = θ̂[1:pμ]; βσ = θ̂[pμ+1:pμ+pσ]; α = θ̂[pμ+pσ+1:end]
        ημ = Xμ * βμ; ησ = Xσ * βσ; ησa = Zg * α
        σa = exp.(ησa)
        Σa = (σa * σa') .* Ksym
        Vm = Matrix{Float64}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            Vm[i, j] = Σa[gidx[i], gidx[j]]
        end
        @inbounds for i in 1:n
            Vm[i, i] += exp(2 * ησ[i])
        end
        Vfac = cholesky(Symmetric(Vm); check = false)
        r = y .- ημ
        Vinv_r = Vfac \ r
        ZtVinv_r = zeros(G)
        for i in 1:n
            ZtVinv_r[gidx[i]] += Vinv_r[i]
        end
        Σa * ZtVinv_r
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
end

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
                        Vm[i, j] += σa[c.gidx[i]]^2
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
                        Vm[i, j] += σa[c.gidx[i]]^2
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
    # Same flag as the scalar route: x/f/g OR, not the iteration cap, and the
    # Newton step in SE units. `g_converged` stays out (absolute 1e-8).
    converged = Optim.converged(res) && !Optim.iteration_limit_reached(res) &&
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
            throw(ArgumentError("drm: sd() supports random INTERCEPT terms only."))
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
        gidx, G = _group_index(getproperty(data, grp))
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
