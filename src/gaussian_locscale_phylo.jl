# gaussian_locscale_phylo.jl — univariate Gaussian location-scale model with
# phylogenetic random effects on BOTH axes (B1 of the σ-phylo plan).
#
# The `:gaussian_mean` leaf from rich_bivariate.jl is a full location-scale leaf:
#   nll = 0.5 * (r²/σ² + log(2π σ²)),   σ = exp(ψ)
#   gη = -r/σ²,   gψ = 1 - r²/σ²
#   hηη = 1/σ²,   hηψ = 2r/σ²,   hψψ = 2r²/σ²
# This means the ENTIRE existing q=2 location-scale Laplace machinery (inner
# mode, marginal, gradient, fit) works for a Gaussian response by passing
# `Val(:gaussian_mean)` as the `kind` argument — no new kernel code needed.
#
# SEPARATE block (MUST-HAVE — the capability drmTMB lacks):
#   Λ = diag(L11², L22²),  L21 ≡ 0  →  mean-phylo RE ⊥ σ-phylo RE.
# COUPLED block (secondary option):
#   Free L21 in the 2×2 Cholesky → mean↔σ correlation in the group-level Λ.
#
# The two blocks use the SAME `_fit_locscale` engine: SEPARATE is recovered by
# fixing L21 = 0 in the initial guess and letting the optimiser relax only
# logL11 and logL22 (via a constrained wrapper that pins L21 = 0); COUPLED lets
# all three λ parameters move freely.
#
# ASYMMETRIC (σ-phylo only, mean fixed) case:
#   Zη = zeros(n, 2), Zψ = [1 0] per row — same as `_sigma_re_loadings` but
#   with a phylogenetic Q instead of Q = I.  The mean-axis Λ diagonal is pinned
#   to a tiny ε = _SIGMA_RE_EPS; only logL22 (= log τ_σ) is optimised.
#   This is the "asymmetric univariate" route (no mean phylo RE).

using SparseArrays: sparse, nonzeros
using LinearAlgebra: I, Symmetric, cholesky, issuccess, diag, logdet, norm, svdvals
import Optim

# ---------------------------------------------------------------------------
# Λ parameterisations for SEPARATE and COUPLED blocks.
# ---------------------------------------------------------------------------

# SEPARATE (diagonal): Λ = diag(L11², L22²), λ = [logL11, logL22] (2-vector).
function _glsp_sep_Λ(λ)
    L11 = exp(λ[1]); L22 = exp(λ[2])
    return [L11^2 0.0; 0.0 L22^2]
end

# COUPLED (free L21): Λ = L Lᵀ, λ = [logL11, L21, logL22] (3-vector).
# Delegates to the existing `_ls_lc_to_Λ` parameterisation in locscale_inner.jl.
_glsp_coupled_Λ(λ) = _ls_lc_to_Λ(λ)   # [logL11, L21, logL22] → Λ

# ---------------------------------------------------------------------------
# Separate-block fitter (the MUST-HAVE, 2 free variance params: logL11, logL22).
# ---------------------------------------------------------------------------

# Marginal NLL at θ = [βμ; βψ; logL11; logL22] with L21 ≡ 0.
function _glsp_sep_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ;
                       warm::Union{Nothing,Ref{Union{Nothing,Vector{Float64}}}} = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    βμ = @view θ[1:pμ]; βψ = @view θ[pμ+1:pμ+pψ]
    λ   = θ[pμ+pψ+1:pμ+pψ+2]   # [logL11, logL22]
    Λ   = _glsp_sep_Λ(λ)
    Λinv = _ls_inv2x2(Λ)
    P = prior_precision(Q, Λinv)
    a0 = warm === nothing ? nothing : warm[]
    val, a, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ; a0 = a0)
    warm !== nothing && ok && (warm[] = copy(a))
    return ok ? val : 1e18
end

# Exact gradient at θ = [βμ; βψ; logL11; logL22].  Recovered from the general
# 5-component gradient (over [βμ; βψ; logL11, L21, logL22]) by embedding and
# collapsing: L21 ≡ 0 is pinned, so dM/d(L21) is never emitted; the logL11 and
# logL22 components are extracted at positions pμ+pψ+1 and pμ+pψ+3.
function _glsp_sep_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ;
                        warm::Union{Nothing,Ref{Union{Nothing,Vector{Float64}}}} = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    λ  = θ[pμ+pψ+1:pμ+pψ+2]   # [logL11, logL22]
    θ_full = vcat(θ[1:pμ+pψ], λ[1], 0.0, λ[2])   # embed L21 = 0
    a0 = warm === nothing ? nothing : warm[]
    g_full = _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ_full, Zη, Zψ; a0 = a0)
    # g_full: [βμ(pμ); βψ(pψ); logL11; L21; logL22]
    # Extract [βμ; βψ; logL11; logL22], drop L21 (index pμ+pψ+2).
    grad = zeros(pμ + pψ + 2)
    grad[1:pμ+pψ]     .= g_full[1:pμ+pψ]
    grad[pμ+pψ+1]      = g_full[pμ+pψ+1]   # logL11
    grad[pμ+pψ+2]      = g_full[pμ+pψ+3]   # logL22 (skip L21)
    return grad
end

# REML (Patterson–Thompson) penalty for integrating out the mean fixed effects β_μ.
# By the marginal-Hessian identity, S = ∂²nll_marginal/∂β_μ² equals the Schur
# complement H_ββ − H_βuᵀ H_uu⁻¹ H_βu — the marginal information of β_μ. We form S
# by finite-differencing the route's analytic β_μ-gradient block (positions 1:pμ),
# and return the restricted correction 0.5·logdet(S). ML stays the default; REML is
# opt-in (method = :REML), reducing the n→n−pμ downward bias in the variance comps.
function _glsp_reml_penalty(grad_fn, θ, pμ::Int; h::Real = 1e-3)
    S = zeros(pμ, pμ)
    for j in 1:pμ
        θp = copy(θ); θp[j] += h
        θm = copy(θ); θm[j] -= h
        gp = grad_fn(θp); gm = grad_fn(θm)
        @views S[:, j] .= (gp[1:pμ] .- gm[1:pμ]) ./ (2h)
    end
    S .= 0.5 .* (S .+ S')                        # symmetrise FD asymmetry
    ch = cholesky(Symmetric(S); check = false)
    # β_μ information not PD (a degenerate / near-boundary point) ⇒ a LARGE FINITE penalty, not
    # Inf: Inf makes the composite gradient non-finite and trips Optim's line-search assertion
    # (a real crash near σ→0). A large finite value lets the line search backtrack away instead.
    issuccess(ch) || return 1e18
    return sum(log, diag(ch.U))                 # = 0.5·logdet(S)
end

# REML observed-information covariance (issue #310). The reported Wald vcov under
# method = :REML must be the inverse Hessian of the RESTRICTED objective
# `nll_ML + 0.5·logdet S`, evaluated at θ̂_reml — NOT the ML observed information
# (which omits the restricted-penalty curvature).
#
#     H_R = H_ML + ∂²(0.5·logdet S)/∂θ².
#
# H_ML is the CLEAN FD Jacobian of the ANALYTIC ML gradient (same as the ML path).
# The penalty Hessian is a second central difference of the smooth penalty VALUE
# `0.5·logdet S(θ)` (accurate to ~1e-8, so a value-based second difference is far
# cleaner than differencing an FD-of-penalty gradient, which would be a noisy
# third-order stencil — see `_glsp_reml_refit`'s note). A larger step `hp` tames the
# O(hp²) second-derivative noise. We add the penalty curvature only to the VARIANCE
# block (indices > pμ+pψ is unknown here, so we symmetrise over all θ and rely on the
# ML block dominating the mean/scale part). Same PD-guard as the ML path: at a
# variance boundary H_R is singular and we return NaN SEs (use profile_ci there).
function _glsp_reml_vcov(grad_fn, θ̂, pμ::Int; h::Real = 1e-4, hp::Real = 1e-2)
    np = length(θ̂)
    # --- H_ML: clean FD of the analytic ML gradient (as the ML path). ---
    Hml = zeros(np, np)
    for j in 1:np
        tp = copy(θ̂); tp[j] += h; tm = copy(θ̂); tm[j] -= h
        Hml[:, j] .= (grad_fn(tp) .- grad_fn(tm)) ./ (2h)
    end
    Hml .= 0.5 .* (Hml .+ Hml')                       # symmetrise FD asymmetry
    # --- Penalty Hessian: second central differences of the penalty VALUE. ---
    penalty(θ) = _glsp_reml_penalty(grad_fn, θ, pμ)
    _finite(v) = isfinite(v) && v < 1e16              # skip the non-PD-S sentinel
    Hpen = zeros(np, np)
    if pμ > 0
        p0 = penalty(θ̂)
        if _finite(p0)
            # Diagonal: (p(θ+e) − 2p(θ) + p(θ−e)) / hp².
            pplus = zeros(np); pminus = zeros(np)
            for i in 1:np
                θp = copy(θ̂); θp[i] += hp; θm = copy(θ̂); θm[i] -= hp
                pplus[i] = penalty(θp); pminus[i] = penalty(θm)
                (_finite(pplus[i]) && _finite(pminus[i])) &&
                    (Hpen[i, i] = (pplus[i] - 2p0 + pminus[i]) / hp^2)
            end
            # Off-diagonals: the mixed second difference (i≠j), symmetrised.
            for i in 1:np, j in (i+1):np
                θpp = copy(θ̂); θpp[i] += hp; θpp[j] += hp
                θmm = copy(θ̂); θmm[i] -= hp; θmm[j] -= hp
                vpp = penalty(θpp); vmm = penalty(θmm)
                if _finite(vpp) && _finite(vmm) && _finite(pplus[i]) &&
                   _finite(pplus[j]) && _finite(pminus[i]) && _finite(pminus[j])
                    hij = (vpp - pplus[i] - pplus[j] + 2p0 - pminus[i] - pminus[j] + vmm) /
                          (2 * hp^2)
                    Hpen[i, j] = hij; Hpen[j, i] = hij
                end
            end
        end
    end
    H = Hml .+ Hpen
    chH = cholesky(Symmetric(H); check = false)
    return issuccess(chH) ? Matrix(inv(chH)) : fill(NaN, np, np)
end

# REML re-fit: starting from the ML estimate θ̂_ml, minimise nll_REML = nll_ML + the
# Patterson–Thompson penalty over θ. Returns (θ̂, converged, ml_nll, reml_nll).
#
# The penalty is a logdet of a FINITE-DIFFERENCE-formed β_μ-information, so its own
# gradient is a second finite difference — too noisy for LBFGS's gradient-convergence
# flag to fire near the (flat) variance-component optimum (a false negative: the search
# parks at the right θ̂ but Optim reports `converged = false`). The re-fit is a small
# polish of the ALREADY-converged ML estimate, so judge convergence on substance, not on
# that noisy flag: the ML fit converged, the restricted objective is finite (β_μ-info PD
# ⇒ non-degenerate), and θ̂ stayed in a neighbourhood of θ̂_ml (no runaway to a boundary).
# A clean flag is restored by `_glsp_reml_refit_clean` (single clean penalty FD); this older
# variant is kept only as the FD-REML correctness anchor for the tests.
function _glsp_reml_refit(obj, grad_fn, θ̂_ml, pμ::Int; ml_converged::Bool = true)
    reml_obj(θ) = obj(θ) + _glsp_reml_penalty(grad_fn, θ, pμ)
    # LBFGS with a finite-difference gradient, warm-started from the ML estimate (the REML
    # optimum is nearby) — a handful of steps, far cheaper than the hundreds NelderMead
    # burns near a flat variance-component optimum.
    # Guard the line search (same fragility as _glsp_reml_refit_clean): on weak/boundary data a
    # probe can hit the non-PD-penalty cliff and HagerZhang asserts; on failure fall back to the
    # ML estimate with converged=false instead of throwing out of drm().
    res = try
        Optim.optimize(reml_obj, θ̂_ml, Optim.LBFGS(),
                       Optim.Options(g_tol = 1e-3, iterations = 100))
    catch err
        err isa InterruptException && rethrow(err)
        nothing
    end
    res === nothing &&
        return copy(θ̂_ml), false, obj(θ̂_ml), (let r = reml_obj(θ̂_ml); isfinite(r) && r < 1e16 ? r : NaN end)
    θ̂ = Optim.minimizer(res)
    reml_nll = reml_obj(θ̂)
    isfinite(reml_nll) && reml_nll ≥ 1e16 && (reml_nll = NaN)   # 1e18 penalty sentinel ⇒ NaN
    converged = ml_converged && isfinite(reml_nll) && norm(θ̂ .- θ̂_ml) < 5.0
    return θ̂, converged, obj(θ̂), reml_nll
end

# Clean-gradient REML refit — the Patterson–Thompson composite REML used by the Poisson
# and sparse-Laplace GLMM routes. (It WAS the σ-phylo location-scale REML until Arc 2;
# those routes now use `_glsp_joint_reml_fit`, native drmTMB's joint-Laplace quantity.)
# Jointly optimises the restricted objective `nll_ML + 0.5·logdet S` over ALL of θ (so it is
# jointly stationary, unlike the block-coordinate Newton) with a CLEAN gradient (exact ML
# gradient + a single FD of the accurate penalty), and is boundary-robust (finite penalty +
# guarded line search). The observed-information Newton (`_glsp_reml_newton`) is faster on
# benign data but the adversarial verification (2026-06-12) found a β-coupling bias at larger
# pμ/pψ and boundary issues, so it is EXPERIMENTAL — this is the wired path.
# (Earlier mis-called "AI-REML stage 1"; the literal average-information data-quadratic was
# proven invalid for this augmented-Laplace model — see `_glsp_reml_newton`.)
#
# `_glsp_reml_refit` lets LBFGS finite-difference the whole composite `nll_ML + 0.5·logdet
# S`. Because S is itself an FD of the β_μ-gradient block, that is a SECOND-order finite
# difference — too noisy for the gradient-convergence flag, so the FD refit judged
# convergence on substance. Here the optimiser gets a CLEAN gradient: the EXACT analytic ML
# gradient (`grad_fn`) plus a SINGLE central FD of the penalty VALUE (logdet S is smooth and
# accurate to ~1e-8, so its single FD is clean). Same restricted objective ⇒ same optimum as
# the FD refit, but the flag fires honestly and the search is a handful of steps.
#
# Returns (θ̂, converged, ml_nll, reml_nll, n_steps).
function _glsp_reml_refit_clean(obj, grad_fn, θ̂_ml, pμ::Int; ml_converged::Bool = true)
    penalty(θ)  = _glsp_reml_penalty(grad_fn, θ, pμ)
    reml_obj(θ) = obj(θ) + penalty(θ)
    function reml_grad!(g, θ)
        h = 1e-4
        g .= grad_fn(θ)                                  # exact analytic ML gradient
        @inbounds for j in eachindex(θ)
            θp = copy(θ); θp[j] += h
            θm = copy(θ); θm[j] -= h
            g[j] += (penalty(θp) - penalty(θm)) / (2h)   # clean single-FD of the penalty
        end
        return g
    end
    # Guard the line search: near the variance boundary a probe can still hit a non-finite
    # value; on failure fall back to the ML estimate (the n→n−pμ REML correction is negligible
    # at the σ→0 boundary). Without this guard, drm(method=:REML) crashed on ~5–8% of
    # near-boundary datasets with an opaque LineSearches AssertionError (verification 2026-06-12).
    res = try
        Optim.optimize(reml_obj, reml_grad!, copy(θ̂_ml), Optim.LBFGS(),
                       Optim.Options(g_tol = 1e-3, iterations = 100))
    catch err
        err isa InterruptException && rethrow(err)
        nothing
    end
    # `_glsp_reml_penalty` returns a 1e18 sentinel when the β_μ information is non-PD (a
    # collinear / rank-deficient mean design); map a sentinel-poisoned objective to NaN so the
    # isfinite guards below AND downstream loglik/AIC/BIC catch it instead of reporting a
    # poisoned −1e18 (verification 2026-06-12). `_sane` ≡ finite and below the sentinel scale.
    _sane(r) = (isfinite(r) && r < 1e16) ? r : NaN
    res === nothing && return copy(θ̂_ml), false, obj(θ̂_ml), _sane(reml_obj(θ̂_ml)), 0
    θ̂ = Optim.minimizer(res)
    reml_nll = _sane(reml_obj(θ̂))
    n_steps = Optim.iterations(res)
    # SUBSTANCE-based flag: the FD-penalty gradient's noise floor and the n-scaling ML gradient
    # make Optim's absolute g_tol unreliable (it reported converged=false on correct fits at
    # larger n — verification finding). Judge on substance: ML converged, restricted objective
    # finite, θ̂ near θ̂_ml (no runaway). A boundary solution (θ̂ at the σ→0 edge) is a valid
    # optimum; its SE is handled by the Wald-V guard / profile CI downstream.
    converged = ml_converged && isfinite(reml_nll) && norm(θ̂ .- θ̂_ml) < 5.0
    return θ̂, converged, obj(θ̂), reml_nll, n_steps
end

# (Unused since Arc 2; was the σ-phylo REML) a fast observed-information Newton WARM START, then a
# clean-gradient LBFGS POLISH that guarantees joint stationarity — this corrects the Newton's
# block-coordinate β-coupling (the polish matches FD-REML to <0.01% even at pμ=6/pψ=2, where the
# bare Newton drifts ~0.7%). Falls back to the STABLE FD-REML if the polished fit fails to
# converge. So the default is fast (Newton-warmed) AND jointly correct AND has a stable backstop.
# Returns (θ̂, converged, ml_nll, reml_nll).
function _glsp_reml_fit(obj, grad, θ̂_ml, pμ::Int, vidx::AbstractVector{Int}; ml_converged::Bool = true)
    θw, convw, _, _, _ = _glsp_reml_newton(obj, grad, θ̂_ml, pμ, vidx; ml_converged = ml_converged)
    start = (convw && all(isfinite, θw)) ? θw : θ̂_ml          # warm-start the polish from the Newton if sane
    θ̂, conv, ml_nll, reml_nll, _ = _glsp_reml_refit_clean(obj, grad, start, pμ; ml_converged = ml_converged)
    if !(conv && isfinite(reml_nll))                          # stable fallback: FD-REML from the ML estimate
        θ̂, conv, ml_nll, reml_nll = _glsp_reml_refit(obj, grad, θ̂_ml, pμ; ml_converged = ml_converged)
    end
    return θ̂, conv, ml_nll, reml_nll
end

# EXPERIMENTAL (asymmetric σ-phylo route) — NOT the production REML (that is the jointly-correct
# `_glsp_reml_refit_clean`). This is the older NON-backtracking observed-information Newton;
# prefer the general `_glsp_reml_newton` (safeguarded step). Same β-coupling / boundary caveats
# as the general one (verification 2026-06-12). Kept only for the asymmetric characterisation test.
#
# HONEST FINDING (adversarial derivation panel, 2026-06-12): the textbook average-information
# DATA QUADRATIC — AI = ½ (∂P_k â)ᵀ H⁻¹ (∂P_l â) — is INVALID as the Newton metric for this
# augmented-Laplace model. The GTC identity E[(∂P â)ᵀH⁻¹(∂P â)] = tr(H⁻¹∂P H⁻¹∂P) holds only
# in EXPECTATION over â ~ N(0, H⁻¹); the realised inner mode â is the SHRUNK BLUP (covariance
# ≠ H⁻¹), so the realised quadratic lands at ~0.2× of the trace (measured: 10.0 vs the
# observed Hessian 22.2 vs the expected-info trace 51.8). With that ~5×-too-small curvature
# the AI-Newton diverges (20 steps, 18% off). The correct, FASTEST O(p) metric is the
# OBSERVED information: a central FD of the EXACT O(p) marginal + penalty REML score (each
# score eval is O(p) — analytic gradient + Takahashi + one clean penalty FD). For K variance
# components this is O(Kp) and converges in a handful of Newton steps (measured, not a
# guaranteed bound — and only on benign data; see the EXPERIMENTAL caveats above). (The expected-info trace
# ½ tr(H⁻¹∂P_k H⁻¹∂P_l) is also valid but needs off-pattern H⁻¹ via factored solves and is
# slower — 14 steps; the observed-info FD is simpler and faster.)
#
# Block-coordinate, ASReml-like: conditional fixed-effect re-fit, then an observed-info Newton
# step on logL22. Returns (θ̂, converged, ml_nll, reml_nll, n_newton).
function _glsp_reml_newton_asym(kind, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ, θ̂_ml, pμ::Int;
                                ml_converged::Bool = true, tol::Real = 1e-4, maxit::Int = 12)
    pψ = size(Xψ, 2)
    iσ = length(θ̂_ml)          # logL22 is the last entry
    iβ = 1:(pμ + pψ)
    θ = copy(θ̂_ml)
    obj(t)  = _glsp_asym_nll(kind, y, Xμ, Xψ, gidx, G, Q, t, Zη, Zψ)
    grd(t)  = _glsp_asym_grad(kind, y, Xμ, Xψ, gidx, G, Q, t, Zη, Zψ)
    pen(t)  = _glsp_reml_penalty(grd, t, pμ)
    # clean REML score for logL22: exact ML grad + single central FD of the accurate penalty.
    function score_σ(t; h = 1e-4)
        tp = copy(t); tp[iσ] += h
        tm = copy(t); tm[iσ] -= h
        return grd(t)[iσ] + (pen(tp) - pen(tm)) / (2h)
    end
    # OBSERVED information = central FD of the clean score w.r.t. logL22 (β at its conditional
    # optimum). The PD curvature for a Newton step; O(p) per score eval (the panel's 3-step winner).
    function curv_σ(t; h = 1e-3)
        tp = copy(t); tp[iσ] += h
        tm = copy(t); tm[iσ] -= h
        return (score_σ(tp) - score_σ(tm)) / (2h)
    end
    # conditional re-fit of the fixed effects (βμ, βψ) holding logL22 fixed.
    function refit_β!(t)
        ls = t[iσ]
        ob(b) = obj(vcat(b, ls))
        gb!(g, b) = (g .= grd(vcat(b, ls))[iβ]; g)
        r = Optim.optimize(ob, gb!, t[iβ], Optim.LBFGS(), Optim.Options(g_tol = 1e-7, iterations = 100))
        t[iβ] .= Optim.minimizer(r)
        return t
    end
    n_newton = 0; converged = false
    for it in 1:maxit
        n_newton = it
        refit_β!(θ)                               # (a) GLS-like fixed-effect update
        s = score_σ(θ)
        if abs(s) < tol                           # (b) observed-info Newton on logL22
            converged = true
            break
        end
        H = curv_σ(θ)
        isfinite(H) || break
        Hpd = abs(H) < 1e-6 ? 1e-6 : abs(H)       # project to PD (descent) + guard a flat curvature
        θ[iσ] -= clamp(s / Hpd, -3.0, 3.0)        # clamp IS the trust bound when curvature is poor
    end
    refit_β!(θ)                                   # final conditional fixed-effect polish
    ml_nll = obj(θ); reml_nll = ml_nll + pen(θ)
    isfinite(reml_nll) && reml_nll ≥ 1e16 && (reml_nll = NaN)   # 1e18 penalty sentinel ⇒ NaN (collinear mean)
    converged = converged && ml_converged && isfinite(reml_nll)
    return θ, converged, ml_nll, reml_nll, n_newton
end

# EXPERIMENTAL — NOT the production REML path (the wired path is `_glsp_reml_refit_clean`).
# The adversarial verification (2026-06-12) confirmed two defects in this block-coordinate
# Newton: (1) `refit_β!` minimises ONLY the ML objective over β, omitting the penalty's
# β-dependence, so the fixed point is NOT jointly stationary — a σ-SD bias that is negligible
# for intercept-ish models (pμ≈pψ≈1, Δsd≈5e-6) but grows with pμ/pψ (≈3% at pμ=6, pψ=2); and
# (2) the variance-block convergence test can mis-fire at the τ→0 boundary. Kept for research /
# the speed exploration. On the verified benign fixtures it matches FD-REML in a handful of
# Newton steps (≤5 asym / ≤6 K=2, measured on seeds 808/909 — not a guaranteed bound). Per-step
# cost is O(K²·pμ) exact O(p) score/penalty evals for the curvature PLUS one conditional
# fixed-effect LBFGS refit: O(p) in the number of groups, constant growing in K and pμ.
#
# General observed-information Newton REML over K variance components (`vidx` = their indices
# in θ). Route-AGNOSTIC: it needs only the route's marginal NLL `obj` and exact gradient `grad`
# (closures over the full θ), so the same code serves the asymmetric (K=1), separate (K=2), and
# coupled (K=3) blocks. The metric is the OBSERVED information — a central FD of the clean
# REML score (exact ML grad + a single FD of the accurate penalty) — projected to PD; the
# average-information data-quadratic is invalid here (â is the shrunk BLUP). Block-coordinate:
# conditional fixed-effect re-fit, then a K×K Newton step on θ[vidx]. Returns
# (θ̂, converged, ml_nll, reml_nll, n_newton).
function _glsp_reml_newton(obj, grad, θ̂_ml, pμ::Int, vidx::AbstractVector{Int};
                           ml_converged::Bool = true, tol::Real = 1e-4, maxit::Int = 20)
    K  = length(vidx)
    nθ = length(θ̂_ml)
    βidx = setdiff(1:nθ, vidx)
    θ  = copy(θ̂_ml)
    pen(t) = _glsp_reml_penalty(grad, t, pμ)
    # clean REML score on the variance components: exact ML grad + single FD of the penalty.
    function score_v(t; h = 1e-4)
        gv = grad(t)[vidx]
        @inbounds for (c, j) in enumerate(vidx)
            tp = copy(t); tp[j] += h
            tm = copy(t); tm[j] -= h
            gv[c] += (pen(tp) - pen(tm)) / (2h)
        end
        return gv
    end
    # observed information (K×K) = central FD of the variance-component score.
    function info_v(t; h = 1e-3)
        A = zeros(K, K)
        @inbounds for (c, j) in enumerate(vidx)
            tp = copy(t); tp[j] += h
            tm = copy(t); tm[j] -= h
            A[:, c] .= (score_v(tp) .- score_v(tm)) ./ (2h)
        end
        return 0.5 .* (A .+ A')                # symmetrise FD asymmetry
    end
    # PD-projected solve: floor eigenvalues so the Newton step is a descent step.
    function pd_solve(A, b)
        E = eigen(Symmetric(A))
        d = max.(E.values, 1e-6)
        return E.vectors * ((E.vectors' * b) ./ d)
    end
    # conditional re-fit of the fixed effects holding the variance components fixed.
    function refit_β!(t)
        ob(b)     = (tw = copy(t); tw[βidx] .= b; obj(tw))
        gb!(g, b) = (tw = copy(t); tw[βidx] .= b; g .= grad(tw)[βidx]; g)
        r = Optim.optimize(ob, gb!, t[βidx], Optim.LBFGS(), Optim.Options(g_tol = 1e-7, iterations = 100))
        t[βidx] .= Optim.minimizer(r)
        return t
    end
    n_newton = 0; converged = false
    for it in 1:maxit
        n_newton = it
        refit_β!(θ)                                   # (a) conditional fixed-effect update
        s = score_v(θ)
        A = info_v(θ)
        all(isfinite, A) || break
        dir = clamp.(pd_solve(A, s), -4.0, 4.0)        # Newton direction (per-component cap)
        f0  = obj(θ) + pen(θ)
        # (b) SAFEGUARDED observed-info Newton step: backtrack until the restricted objective
        # DECREASES (monotone descent ⇒ the iterate cannot diverge even when the FD curvature is
        # unreliable at large p — the unguarded step overshot to σ-SD ≈ 7; benchmark 2026-06-12).
        # Convergence is judged on the STEP SIZE in the log-SD chart (scale-free), NOT on the raw
        # score (which scales with n and made the absolute tol unreachable at large p).
        α = 1.0; accepted = false; δθ = 0.0
        for _ in 1:24
            θt = copy(θ)
            @inbounds for c in 1:K
                θt[vidx[c]] -= α * dir[c]
            end
            refit_β!(θt)                              # β at its conditional optimum for the trial
            if obj(θt) + pen(θt) < f0
                δθ = α * norm(dir); copyto!(θ, θt); accepted = true; break
            end
            α *= 0.5
        end
        if !accepted || δθ < tol                      # no further descent / negligible step ⇒ converged
            converged = true
            break
        end
    end
    refit_β!(θ)
    ml_nll = obj(θ); reml_nll = ml_nll + pen(θ)
    isfinite(reml_nll) && reml_nll ≥ 1e16 && (reml_nll = NaN)   # 1e18 penalty sentinel ⇒ NaN (collinear mean)
    converged = converged && ml_converged && isfinite(reml_nll)
    return θ, converged, ml_nll, reml_nll, n_newton
end

# ---------------------------------------------------------------------------
# JOINT-LAPLACE REML — the restricted likelihood native drmTMB optimises (Arc 2).
# ---------------------------------------------------------------------------
# When `sigma` carries a phylogenetic variance component, native drmTMB's
# `REML = TRUE` (`drm_apply_estimator_spec()`) puts `beta_mu` AND `beta_sigma`
# into TMB's random vector next to the phylogenetic effects, so its outer
# objective is ONE Laplace approximation over z = (a, β) jointly, with a flat
# prior on β:
#
#   nll_R(v) = jn(â, β̂) + ½ logdet H_zz − ½ logdet P − (p/2) log 2π,
#
# where v are the variance parameters only (the Λ parameterisation), (â, β̂) is
# the JOINT mode of jn(a, β) = Σᵢ nllᵢ(ηᵢ, ψᵢ) + ½ aᵀPa at fixed v, H_zz its full
# Hessian, and p = pμ + pψ. The −(p/2) log 2π is TMB's Laplace constant for the p
# flat-prior coordinates. Because logdet H_zz = logdet H_aa + logdet S, with
# S = H_ββ − H_βa H_aa⁻¹ H_aβ the Schur complement (= the Hessian of the profile
# h(β) = min_a jn(a, β)), the value is evaluated with the existing sparse inner
# solve plus a p×p Schur complement.
#
# This is NOT the Patterson–Thompson composite `_glsp_reml_refit_clean`
# minimises (nll_ML(β, v) + ½ logdet ∂²nll_ML/∂β², optimised over β AND v): that
# differs in (i) where β sits (the mode of the Laplace MARGINAL, not the joint
# mode), (ii) which Hessian S is (the marginal's, which carries the β_ψ
# dependence of logdet H_aa, not the joint Schur complement) and (iii) the
# constant. On the Arc 1 probe fixture the old objective reported −146.1213
# against native's −143.3750; this one reproduces native (receipt:
# docs/dev-log/evidence/arc2-gaussian-sigma-phylo-reml/). The fixed effects
# reported under REML are the joint mode β̂(v̂) — what TMB returns for a
# random-vector coordinate.

# Data-part β gradient, β Hessian and the a–β cross Hessian at (a, β).
function _glsp_joint_beta_blocks(kind, y, Xμ, Xψ, gidx, G, a, η0, ψ0, Zη, Zψ)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2); p = pμ + pψ
    gβ = zeros(p); Hββ = zeros(p, p); Haβ = zeros(2G, p)
    @inbounds for i in eachindex(y)
        g = gidx[i]
        a1 = a[2g-1]; a2 = a[2g]
        ηi = η0[i] + Zη[i, 1] * a1 + Zη[i, 2] * a2
        ψi = ψ0[i] + Zψ[i, 1] * a1 + Zψ[i, 2] * a2
        gη, gψ = _ls_grad(kind, y[i], ηi, ψi)
        hηη, hηψ, hψψ = _ls_hess(kind, y[i], ηi, ψi)
        for j in 1:pμ
            xj = Xμ[i, j]
            gβ[j] += gη * xj
            for k in 1:pμ;  Hββ[j, k]      += hηη * xj * Xμ[i, k]; end
            for k in 1:pψ;  Hββ[j, pμ+k]   += hηψ * xj * Xψ[i, k]; end
            Haβ[2g-1, j] += (hηη * Zη[i, 1] + hηψ * Zψ[i, 1]) * xj
            Haβ[2g,   j] += (hηη * Zη[i, 2] + hηψ * Zψ[i, 2]) * xj
        end
        for j in 1:pψ
            xj = Xψ[i, j]
            gβ[pμ+j] += gψ * xj
            for k in 1:pψ;  Hββ[pμ+j, pμ+k] += hψψ * xj * Xψ[i, k]; end
            Haβ[2g-1, pμ+j] += (hηψ * Zη[i, 1] + hψψ * Zψ[i, 1]) * xj
            Haβ[2g,   pμ+j] += (hηψ * Zη[i, 2] + hψψ * Zψ[i, 2]) * xj
        end
    end
    for j in 1:pμ, k in 1:pψ
        Hββ[pμ+k, j] = Hββ[j, pμ+k]
    end
    return gβ, Hββ, Haβ
end

# Joint mode (â, β̂) of jn at fixed P, by Newton on the profile h(β) = min_a jn(a, β)
# (gradient ∂jn/∂β at â(β) by the envelope theorem, Hessian the Schur complement S),
# with a backtracking line search on h and a warm-started inner solve per trial. The
# nested loop only needs to reach the neighbourhood of the mode (its β-gradient
# floor is set by the inner solve's own 1e-9 tolerance); two FULL joint Newton
# steps on (a, β) through the bordered system then polish both coordinates
# together, so the restricted NLL is smooth in P to rounding — the outer
# optimiser differences it. Returns (nll_R, β̂, â, S, ok).
function _glsp_joint_reml_nll(kind, y, Xμ, Xψ, gidx, G, P, Zη, Zψ, β0, a0;
                              tol::Real = 1e-7, maxiter::Int = 100,
                              noise_floor::Bool = false)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2); p = pμ + pψ
    chP = cholesky(Symmetric(P); check = false)
    issuccess(chP) || return Inf, β0, a0, nothing, false
    β = copy(β0)
    # `noise_floor` (the coupled block only). Near its |cor| → 1 boundary
    # (logL22 → −∞) the prior precision P = Q ⊗ Λ⁻¹ reaches 1e7 and beyond, and the
    # rounding noise of the joint gradient at the exact mode, ~eps·‖P‖·‖a‖, exceeds
    # the inner solver's absolute 1e-9 stationarity bound. Every inner solve then
    # spins to its iteration cap (~1.5 s) and fails, and nll_R is Inf there (measured:
    # G1 coupled fixture from cor ≈ 1 − 1e-5 outwards; native drmTMB's bound is
    # 1 − 1e-6). The bound is therefore raised to that noise floor, eps·‖P‖, when it
    # exceeds 1e-9. (A 16× looser bound left the mode too rough for the β Newton to
    # converge.) Off for the other blocks: the asymmetric block pins a zero-loading
    # axis at a tiny variance, so its ‖P‖ is huge by construction while its solves
    # are clean.
    tol_in = noise_floor ? max(1e-9, eps(Float64) * maximum(abs, nonzeros(P))) : 1e-9
    # Certified Newton from a warm start. The outer loops re-solve the inner mode
    # from a neighbouring one thousands of times. From there one Newton step takes
    # the gradient to ~1e-7 and the next to ~1e-12, but that last step can raise
    # jn by a few ULPs of rounding, so `_ls_inner_mode`'s monotone line search
    # rejects it, damps to its cap and fails after 1-2 s; the cold fallback then
    # usually succeeds in ~10 ms (measured on the H2 coupled fixture: 17% of warm
    # solves failed this way and took 87% of the REML time, and inside the β line
    # search a run of such failures held one evaluation for minutes). So plain
    # Newton steps come first, and their end point is accepted on the solver's own
    # certificate (stationary to `tol_in`, PD Hessian), provided the gradient
    # contracted at every step and jn ends no higher than at the start beyond
    # rounding. Anything else falls through to the safeguarded solve below.
    function newton_warm(η0, ψ0, astart)
        a = copy(astart); gprev = Inf
        f0 = _ls_joint(kind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
        isfinite(f0) || return nothing
        for _ in 1:8
            g = _ls_joint_grad(kind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
            all(isfinite, g) || return nothing
            gn = norm(g)
            if gn <= tol_in * (1 + norm(a))
                ft = _ls_joint(kind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
                (isfinite(ft) && ft <= f0 + 1e-10 * (1 + abs(f0))) || return nothing
                chc, okc = _ls_inner_certificate(kind, y, η0, ψ0, gidx, G, P, Zη, Zψ, a, tol_in)
                return okc ? (a, chc, ft) : nothing
            end
            gn < gprev || return nothing
            gprev = gn
            chn = _ls_hess_chol(kind, y, η0, ψ0, gidx, G, a, P, Zη, Zψ)
            issuccess(chn) || return nothing
            a = a .- (chn \ g)
            all(isfinite, a) || return nothing
        end
        return nothing
    end
    profile(βt, astart) = begin
        η0 = Xμ * βt[1:pμ]; ψ0 = Xψ * βt[pμ+1:p]
        fast = newton_warm(η0, ψ0, astart)
        fast === nothing || return fast[3], fast[1], fast[2], true, η0, ψ0
        at, ch, ok = _ls_inner_mode(kind, y, η0, ψ0, gidx, G, P, Zη, Zψ; a0 = astart, tol = tol_in)
        # A warm start can land within rounding of the mode and then fail the inner
        # stationarity certificate; a cold solve is the documented fallback.
        ok || ((at, ch, ok) = _ls_inner_mode(kind, y, η0, ψ0, gidx, G, P, Zη, Zψ; tol = tol_in))
        (ok ? _ls_joint(kind, y, η0, ψ0, gidx, at, P, Zη, Zψ) : Inf), at, ch, ok, η0, ψ0
    end
    hval, a, ch, ok, η0, ψ0 = profile(β, a0)
    ok || return Inf, β, a, nothing, false
    converged = false
    for _ in 1:maxiter
        gβ, Hββ, Haβ = _glsp_joint_beta_blocks(kind, y, Xμ, Xψ, gidx, G, a, η0, ψ0, Zη, Zψ)
        if norm(gβ) <= tol * (1 + norm(β))
            converged = true
            break
        end
        S = Symmetric(Hββ .- Haβ' * (ch \ Haβ))
        chS = cholesky(S; check = false)
        step = issuccess(chS) ? (chS \ gβ) : gβ          # steepest descent if S is not PD
        α = 1.0; moved = false
        while α >= 1e-10
            βt = β .- α .* step
            ht, at, cht, okt, η0t, ψ0t = profile(βt, a)
            if okt && isfinite(ht) && ht <= hval
                β, a, ch, hval, η0, ψ0 = βt, at, cht, ht, η0t, ψ0t
                moved = true
                break
            end
            α *= 0.5
        end
        if !moved
            converged = norm(gβ) <= 1e-5 * (1 + norm(β))
            break
        end
    end
    converged || return Inf, β, a, nothing, false
    # Joint Newton polish on z = (a, β): [H_aa H_aβ; H_βa H_ββ] dz = ∇jn, solved by the
    # Schur complement. Kept only while jn does not rise beyond rounding.
    jn0 = hval
    for _ in 1:2
        ga = _ls_joint_grad(kind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
        gβ, Hββ, Haβ = _glsp_joint_beta_blocks(kind, y, Xμ, Xψ, gidx, G, a, η0, ψ0, Zη, Zψ)
        chA = _ls_hess_chol(kind, y, η0, ψ0, gidx, G, a, P, Zη, Zψ)
        issuccess(chA) || break
        Ha_ga = chA \ ga; Ha_Haβ = chA \ Haβ
        chS = cholesky(Symmetric(Hββ .- Haβ' * Ha_Haβ); check = false)
        issuccess(chS) || break
        dβ = chS \ (gβ .- Haβ' * Ha_ga)
        da = Ha_ga .- Ha_Haβ * dβ
        βt = β .- dβ; at = a .- da
        η0t = Xμ * βt[1:pμ]; ψ0t = Xψ * βt[pμ+1:p]
        jt = _ls_joint(kind, y, η0t, ψ0t, gidx, at, P, Zη, Zψ)
        (isfinite(jt) && jt <= jn0 + 1e-9 * (1 + abs(jn0))) || break
        β, a, η0, ψ0, jn0 = βt, at, η0t, ψ0t, jt
    end
    _, Hββ, Haβ = _glsp_joint_beta_blocks(kind, y, Xμ, Xψ, gidx, G, a, η0, ψ0, Zη, Zψ)
    chA = _ls_hess_chol(kind, y, η0, ψ0, gidx, G, a, P, Zη, Zψ)
    issuccess(chA) || return Inf, β, a, nothing, false
    S = Matrix(Symmetric(Hββ .- Haβ' * (chA \ Haβ)))
    chS = cholesky(Symmetric(S); check = false)
    issuccess(chS) || return Inf, β, a, S, false
    jn = _ls_joint(kind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    nll = jn + 0.5 * logdet(chA) + 0.5 * logdet(chS) - 0.5 * logdet(chP) - 0.5 * p * log(2π)
    return nll, β, a, S, true
end

# REML start from an ML variance estimate: pull each log-SD coordinate (`idx`) up to
# at least log(0.05). An ML fit at the variance boundary (SD → 0, or |cor| → 1 in the
# coupled block, where logL22 → −∞) is a poor REML start — the joint mode is
# ill-conditioned there and Newton stalls (measured: F1 coupled ML start at
# logL22 = −5.3 never moved) — and REML variance estimates sit above ML's anyway.
_glsp_reml_start(v, idx) = (w = copy(v); for i in idx; w[i] = max(w[i], log(0.05)); end; w)

# Numerical rank deficiency of a design: smallest/largest singular value of the
# column-normalised matrix below 1e-10.
function _glsp_rank_deficient(X)
    size(X, 2) <= 1 && return size(X, 2) == 1 && all(iszero, X)
    nrm = [norm(c) for c in eachcol(X)]
    any(iszero, nrm) && return true
    sv = svdvals(X ./ nrm')
    return sv[end] / sv[1] < 1e-10
end

# Outer REML fit over the variance parameters v only. `Λfun(v)` is the route's Λ
# parameterisation. Starts from each of `starts` and keeps the lowest restricted
# NLL. The outer gradient and Hessian are central differences of nll_R (the joint
# mode is re-solved to 1e-10, so the difference quotients are clean). Returns a NamedTuple with the REML
# estimates, the joint-mode β̂, the restricted and ML NLLs and a Wald covariance
# of (β, v): V_vv = (∇²nll_R)⁻¹, and the β rows follow TMB's sdreport rule for a
# random-vector coordinate, Var(β̂) = S⁻¹ + J V_vv Jᵀ with J = dβ̂/dv.
#
# `logsd_idx` names the log-SD coordinates of v (the ones that run to −∞ at the
# variance boundary); `profile_idx` names the log-SD coordinates to profile. Each
# profile interval is on the RESTRICTED surface: nll_R with β integrated out, the
# other variance coordinates re-optimised, thresholded from the REML minimum. It is
# returned in `ci` as (sd_lo, sd_hi), in the order of `profile_idx`.
#
# `cor_edge = true` (the coupled block, v = [logL11, L21, logL22]) also fits the
# correlation boundary |cor| = `_GLSP_COR_CAP` as its own candidate. Native drmTMB
# bounds the phylo correlation, rho = 0.999999·tanh(eta_cor_phylo), and when the data
# put the μ and σ phylo effects on one axis its optimum sits on that bound (G1
# fixture: eta_cor_phylo = 9.6). The interior FD Newton cannot get there: P = Q ⊗ Λ⁻¹
# reaches 1e8 and beyond, the prior quadratic in the joint NLL cancels to ~1e-8 of
# rounding noise, and Newton stalled 1.7e-4 short of native (measured, cor =
# 0.99994). On the boundary the fit is a 2-D Newton in w = (logL11, log sd_σ) for each
# sign of the correlation, evaluated in whitened coordinates: a = L·u with
# u ~ N(0, Q⁻¹ ⊗ I) and loadings (Zη·L, Zψ·L). The Laplace approximation is
# invariant to that linear change of the latent variables, so this is the same
# restricted NLL, without the ill-conditioned precision. The bound wins only with an
# nll_R lower by more than 1e-9 relative. It is then flagged as a boundary fit, with no
# Wald covariance (native's standard errors there are NaN too).
function _glsp_joint_reml_fit(kind, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ, Λfun, starts,
                              β_start; se::Bool = true, g_tol::Real = 1e-9,
                              logsd_idx = (), shrink_idx = (), profile_idx = (),
                              cor_edge::Bool = false)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2); p = pμ + pψ
    # A rank-deficient fixed-effect design makes the flat-prior β integral improper:
    # S is singular in exact arithmetic, and a numerically-PD S would give a finite
    # but meaningless restricted logLik. Report it as degenerate (NaN logLik,
    # not converged, NaN covariance) — the contract the σ-phylo REML routes already
    # honour (test_reml_newton_sigma_phylo.jl, collinear-mean regression).
    if _glsp_rank_deficient(Xμ) || _glsp_rank_deficient(Xψ)
        v0 = last(starts); np = p + length(v0)          # the callers' ML-based start
        P0 = prior_precision(Q, _ls_inv2x2(Λfun(v0)))
        ml0, _, ok0 = _ls_marginal_nll(kind, y, Xμ * β_start[1:pμ], Xψ * β_start[pμ+1:p],
                                       gidx, G, P0, Zη, Zψ)
        return (θ = vcat(β_start, v0), v = v0, β = copy(β_start), reml_nll = NaN,
                ml_nll = ok0 ? ml0 : NaN, converged = false, V = fill(NaN, np, np),
                ci = [(sd_lo = NaN, sd_hi = NaN) for _ in profile_idx])
    end
    warm_β = Ref(copy(β_start)); warm_a = Ref(zeros(2G))
    # Bound mode (`cor_edge`): v = [logL11, log sd_σ] at cor = edge_sign[]·cap, in the
    # whitened coordinates (loadings Zη·L, Zψ·L; prior Q ⊗ I).
    edge_mode = Ref(false); edge_sign = Ref(1.0)
    P_edge = cor_edge ? prior_precision(Q, Matrix(1.0I, 2, 2)) : nothing
    edge_v(w) = (sdσ = exp(w[2]); c = edge_sign[] * _GLSP_COR_CAP;
                 [w[1], c * sdσ, w[2] + 0.5 * log1p(-c^2)])
    function model_at(v)
        if edge_mode[]
            ve = edge_v(v)
            L = [exp(ve[1]) 0.0; ve[2] exp(ve[3])]          # Λ = L Lᵀ, as `_glsp_coupled_Λ`
            all(isfinite, L) || return nothing
            return P_edge, Zη * L, Zψ * L
        end
        Λ = Λfun(v)
        all(isfinite, Λ) || return nothing
        P = prior_precision(Q, _ls_inv2x2(Λ))
        all(isfinite, nonzeros(P)) || return nothing
        return P, Zη, Zψ
    end
    function eval_v(v; update::Bool = true)
        m = model_at(v)
        m === nothing && return Inf, warm_β[], warm_a[], nothing, false
        P, Zηv, Zψv = m
        solve(β0, a0) = try
            _glsp_joint_reml_nll(kind, y, Xμ, Xψ, gidx, G, P, Zηv, Zψv, β0, a0;
                                 noise_floor = cor_edge)
        catch err
            err isa InterruptException && rethrow(err)
            (Inf, warm_β[], warm_a[], nothing, false)
        end
        r = solve(warm_β[], warm_a[])
        # A warm start left by a distant evaluation can fail where a cold one succeeds
        # (measured near the coupled block's |cor| → 1 edge, where P is
        # ill-conditioned): retry cold there.
        (r[5] || !cor_edge) || (r = solve(β_start, zeros(2G)))
        if update && r[5]
            warm_β[] = copy(r[2]); warm_a[] = copy(r[3])
        end
        return r
    end
    nllR(v) = (r = eval_v(v); r[5] ? r[1] : 1e18)
    h = 1e-5
    function fdgrad(v)
        g = similar(v, Float64)
        for j in eachindex(v)
            vp = copy(v); vp[j] += h; vm = copy(v); vm[j] -= h
            g[j] = (nllR(vp) - nllR(vm)) / (2h)
        end
        g
    end
    function fdhess(v; hh = 1e-4)
        k = length(v); H = zeros(k, k); fc = nllR(v)
        for i in 1:k
            vp = copy(v); vp[i] += hh; vm = copy(v); vm[i] -= hh
            H[i, i] = (nllR(vp) - 2 * fc + nllR(vm)) / hh^2
            for j in (i+1):k
                vpp = copy(v); vpp[i] += hh; vpp[j] += hh
                vpm = copy(v); vpm[i] += hh; vpm[j] -= hh
                vmp = copy(v); vmp[i] -= hh; vmp[j] += hh
                vmm = copy(v); vmm[i] -= hh; vmm[j] -= hh
                H[i, j] = H[j, i] = (nllR(vpp) - nllR(vpm) - nllR(vmp) + nllR(vmm)) / (4hh^2)
            end
        end
        H
    end
    # Levenberg-damped Newton on nll_R(v) with an FD gradient and FD Hessian: v has
    # at most three coordinates, so each Hessian is ≤ 19 restricted-likelihood
    # evaluations, and Newton converges where an FD-gradient L-BFGS was measured to
    # stall well short of the optimum (F2 coupled fixture, gradient ‖·‖ ≈ 5).
    #
    # Variance boundary. When a log-SD runs to −∞ the restricted NLL flattens onto a
    # plateau: its gradient falls like SD², so near SD ≈ 1e-4 it is ~1e-7 and the FD
    # gradient's rounding noise (~3e-8) keeps it from reaching `g_tol`, while every
    # Newton step still buys ~1e-9. Newton then creeps along the plateau until the
    # iteration cap and reports non-convergence (measured: 9/15 zero-signal
    # sigma-only fits). The fit is declared converged on the boundary once a log-SD
    # coordinate is below `_GLSP_REML_BOUNDARY` (SD < 2.5e-3), the gradient is below
    # the no-descent tolerance used below, and either the last accepted step improved
    # nll_R by at most 1e-10 relative (the objective has stopped moving) or pushing the
    # boundary log-SDs 6 units further out does not raise nll_R (below). The boundary
    # log-SDs are then left on the plateau supremum.
    function newton_min(v0; logsd_idx = logsd_idx, shrink_idx = shrink_idx)
        v = copy(v0); f = nllR(v); gain = Inf
        (isfinite(f) && f < 1e17) || return v, Inf, false
        for _ in 1:100
            g = fdgrad(v)
            all(isfinite, g) || return v, f, false
            norm(g) <= g_tol * (1 + abs(f)) && return v, f, true
            if any(j -> v[j] < _GLSP_REML_BOUNDARY, logsd_idx) &&
               norm(g) <= 1e-5 * (1 + abs(f))
                # Snap each boundary log-SD 6 units further out when that does not
                # raise nll_R: the plateau's supremum is at SD → 0, and the creeping
                # Newton iterate can still sit ~2e-6 below it (measured, coupled block
                # with both SDs at zero).
                # `shrink_idx` are raw-scale coordinates (the coupled block's L21) that
                # also go to zero with a boundary SD; they are tried at e⁻⁶ × their value.
                # The moves interact (a leftover L21 blocks the logL22 snap), so the pass
                # repeats, at most three times, while any move is accepted.
                vs, fs, any_snap = copy(v), f, false
                for _ in 1:3
                    snapped = false
                    for j in shrink_idx
                        vt = copy(vs); vt[j] *= exp(-6.0); ft = nllR(vt)
                        (isfinite(ft) && ft <= fs) && ((vs, fs, snapped) = (vt, ft, true))
                    end
                    for j in logsd_idx
                        vs[j] < _GLSP_REML_BOUNDARY || continue
                        vt = copy(vs); vt[j] -= 6.0; ft = nllR(vt)
                        (isfinite(ft) && ft <= fs) && ((vs, fs, snapped) = (vt, ft, true))
                    end
                    snapped || break
                    any_snap = true
                end
                # Converged on the boundary when the objective has stopped moving, or
                # when an accepted snap shows nll_R does not rise as the SD goes 6
                # log-units further toward zero AND the gradient in the coordinates off
                # the boundary is small. The snap test is what makes the rule robust:
                # the creeping iterate's per-step gain hovers around the 1e-10
                # threshold, so that test alone flipped with platform rounding
                # (zero-signal seed 1011 on Julia 1.10 x86-64, 1019 on aarch64).
                gain <= 1e-10 * (1 + abs(f)) && return vs, fs, true
                if any_snap
                    free = [j for j in eachindex(v) if !(j in logsd_idx && v[j] < _GLSP_REML_BOUNDARY)]
                    (isempty(free) || norm(g[free]) <= 1e-6 * (1 + abs(f))) && return vs, fs, true
                    v, f = vs, fs          # keep the plateau point; refine the rest
                    continue
                end
            end
            H = Symmetric(fdhess(v))
            λ = 0.0; moved = false; scale = 1 + maximum(abs, H)
            while λ <= 1e2 * scale
                ch = cholesky(Symmetric(Matrix(H) + λ * I); check = false)
                if issuccess(ch)
                    d = -(ch \ g)
                    nd = norm(d); nd > 2.0 && (d .*= 2.0 / nd)   # cap a step in log-SD/Cholesky units
                    α = 1.0
                    while α >= 1 / 64
                        vt = v .+ α .* d; ft = nllR(vt)
                        if isfinite(ft) && ft < f
                            gain = f - ft
                            v, f = vt, ft; moved = true
                            break
                        end
                        α *= 0.5
                    end
                    moved && break
                end
                λ = λ == 0.0 ? 1e-4 * scale : 100λ
            end
            moved || return v, f, norm(g) <= 1e-5 * (1 + abs(f))
        end
        return v, f, false
    end
    # `best_warm` keeps the joint mode (β, a) that the winning search last solved,
    # at the optimum or an FD neighbour of it: the fallback start for the final
    # re-evaluation below.
    best_v = nothing; best_f = Inf; best_conv = false; best_warm = nothing
    for v0 in starts
        warm_β[] = copy(β_start); warm_a[] = zeros(2G)
        v, f, conv = newton_min(v0)
        # Native's parameter space stops at |cor| = cap; beyond it lies the boundary
        # candidate's job (below), so an interior iterate past the cap does not count.
        cor_edge && abs(v[2]) / hypot(v[2], exp(v[3])) > _GLSP_COR_CAP && continue
        if f < best_f
            best_v, best_f, best_conv = copy(v), f, conv
            best_warm = (copy(warm_β[]), copy(warm_a[]))
        end
    end
    (best_v === nothing && !cor_edge) &&
        error("REML (joint Laplace): no start produced a finite restricted likelihood")
    # The correlation bound (see the note above the function), for both signs: from SDs of
    # 0.3, and from the best interior iterate's SDs on its own side. An interior fit
    # parked on the sd_μ → 0 plateau (measured: G1, sd_μ = 1.6e-5, cor = −0.45, nll_R
    # 0.97 above native) therefore cannot hide a boundary optimum of either sign.
    edge_w = nothing; best_sign = 1.0
    if cor_edge
        edge_mode[] = true
        vb = best_v === nothing ? [log(0.3), 0.3, log(0.3)] : best_v
        sdσ0 = log(max(hypot(vb[2], exp(vb[3])), 0.05))
        for (sgn, w0) in ((1.0, [log(0.3), log(0.3)]), (-1.0, [log(0.3), log(0.3)]),
                          ((vb[2] < 0 ? -1.0 : 1.0), [max(vb[1], log(0.05)), sdσ0]))
            edge_sign[] = sgn
            warm_β[] = copy(β_start); warm_a[] = zeros(2G)
            w, f, conv = newton_min(w0; logsd_idx = (1, 2), shrink_idx = ())
            # A clear win only: on a both-SDs-zero plateau the boundary and the
            # interior meet, and rounding must not decide between them.
            if f < (isfinite(best_f) ? best_f - 1e-9 * (1 + abs(best_f)) : Inf)
                edge_w, best_f, best_conv, best_sign = copy(w), f, conv, sgn
                best_warm = (copy(warm_β[]), copy(warm_a[]))
            end
        end
        edge_mode[] = edge_w !== nothing
        edge_w === nothing && best_v === nothing &&
            error("REML (joint Laplace): no start produced a finite restricted likelihood")
        if edge_w !== nothing
            edge_sign[] = best_sign
            best_v = edge_v(edge_w)
        end
    end
    eval_at = edge_w === nothing ? best_v : edge_w
    warm_β[] = copy(β_start); warm_a[] = zeros(2G)
    nll_r, β̂, â, S, ok = eval_v(eval_at)
    # The cold start (β_start, a = 0) can fail at the optimum where the search's own
    # warm mode succeeds (measured: H2 fixture, separate block, where the ML β with
    # a = 0 fails at the REML optimum while every point 0.01 away solves). Retry from
    # the mode the search left behind before giving up.
    if !ok && best_warm !== nothing
        warm_β[] = copy(best_warm[1]); warm_a[] = copy(best_warm[2])
        nll_r, β̂, â, S, ok = eval_v(eval_at)
    end
    ok || error("REML (joint Laplace): the joint mode failed at the optimum")
    P̂, Zη̂, Zψ̂ = model_at(eval_at)
    ml_nll, _, ml_ok = _ls_marginal_nll(kind, y, Xμ * β̂[1:pμ], Xψ * β̂[pμ+1:p], gidx, G, P̂, Zη̂, Zψ̂)
    ml_ok || (ml_nll = NaN)
    k = length(best_v); np = p + k
    V = fill(NaN, np, np)
    # On the variance boundary the curvature of nll_R in v is FD rounding noise, so
    # no Wald covariance is reported (NaN) — the convention of the ML routes'
    # PD-guard; `profile_ci = true` gives the boundary-aware interval instead. The
    # correlation bound is a boundary too.
    on_boundary = edge_w !== nothing || any(j -> best_v[j] < _GLSP_REML_BOUNDARY, logsd_idx)
    if se && !on_boundary
        try
            Hv = fdhess(best_v)
            chH = cholesky(Symmetric(Hv); check = false)
            if issuccess(chH)
                Vvv = Matrix(inv(chH))
                J = zeros(p, k); hj = 1e-5
                for j in 1:k
                    vp = copy(best_v); vp[j] += hj; vm = copy(best_v); vm[j] -= hj
                    rp = eval_v(vp; update = false); rm = eval_v(vm; update = false)
                    J[:, j] .= (rp[2] .- rm[2]) ./ (2hj)
                end
                Sinv = Matrix(inv(cholesky(Symmetric(S))))
                V = zeros(np, np)
                V[1:p, 1:p] .= Sinv .+ J * Vvv * J'
                V[1:p, p+1:np] .= J * Vvv
                V[p+1:np, 1:p] .= (J * Vvv)'
                V[p+1:np, p+1:np] .= Vvv
            end
        catch err
            err isa InterruptException && rethrow(err)
            V = fill(NaN, np, np)
        end
    end
    # Restricted-likelihood profile intervals. Every evaluation starts from the joint
    # mode at the REML optimum (update = false), so the bisection's excursions to
    # extreme log-SDs never leave a stale warm start behind; a failed joint mode is
    # Inf, which `_glsp_profile_ci` reads as "not crossed" (the conservative side).
    ci = map(collect(profile_idx)) do j
        # No restricted profile on the correlation bound (the coupled route profiles none).
        edge_w === nothing || return (sd_lo = NaN, sd_hi = NaN)
        warm_β[] = copy(β̂); warm_a[] = copy(â)
        nllP(v) = (r = eval_v(v; update = false); r[5] ? r[1] : Inf)
        gradP(v) = [(vp = copy(v); vp[i] += h; vm = copy(v); vm[i] -= h;
                     (nllP(vp) - nllP(vm)) / (2h)) for i in eachindex(v)]
        _glsp_profile_ci(nllP, gradP, best_v, j)
    end
    return (θ = vcat(β̂, best_v), v = best_v, β = β̂, reml_nll = nll_r, ml_nll = ml_nll,
            converged = best_conv, V = V, ci = ci)
end

# Log-SD below which a REML variance coordinate counts as on the boundary (SD < 2.5e-3).
const _GLSP_REML_BOUNDARY = -6.0

# Native drmTMB's bound on a phylo correlation: rho = 0.999999·tanh(eta) (drmTMB.cpp).
const _GLSP_COR_CAP = 0.999999

# B2 — boundary-aware PROFILE-LIKELIHOOD CI for one variance (log-SD) parameter.
# `nll(θ)::Real` and `grad(θ)::Vector` are the route's own marginal NLL and analytic
# gradient; `idx` is the profiled log-SD position. Profiles θ[idx]: re-optimises the
# free params (cold inner solves — never a shared warm ref) and brackets the χ²₁
# threshold. Boundary-aware: when the profile never crosses the threshold going DOWN
# (logL → −∞, SD → 0), the lower SD endpoint is 0 — an honest `[0, x]` CI instead of a
# singular-Wald failure. Returns (sd_lo, sd_hi) on the SD scale. Generic over the
# block (separate / asymmetric) via the passed closures.
function _glsp_profile_ci(nll, grad, θ̂, idx; level = 0.95)
    nll_min = nll(θ̂)
    thr  = 0.5 * Distributions.quantile(Distributions.Chisq(1), level)   # χ²₁/2
    free = setdiff(1:length(θ̂), idx)
    function prof_dev(v)
        θfix = copy(θ̂); θfix[idx] = v
        # Nothing left to re-optimise (the REML scale-only block: β is integrated
        # out and the SD is the only variance parameter), so the profile is nll itself.
        isempty(free) && return (nll(θfix) - nll_min) - thr
        obj(z) = (θw = copy(θfix); θw[free] .= z; nll(θw))
        grad!(g, z) = (θw = copy(θfix); θw[free] .= z; g .= grad(θw)[free]; g)
        val = try
            res = Optim.optimize(obj, grad!, copy(θ̂[free]), Optim.LBFGS(),
                                 Optim.Options(g_tol = 1e-6, iterations = 150))
            Optim.minimum(res)
        catch
            Inf            # sub-fit failed (ill-conditioned at an extreme log-SD)
        end
        return (val - nll_min) - thr
    end
    crossed(d) = isfinite(d) && d > 0
    v̂ = θ̂[idx]
    # Bounded bracket: cap the log-SD excursion at ±8 (an SD ratio e^8 ≈ 3000×).
    # Beyond that the component is effectively unidentified, so report the boundary
    # (SD 0 below / Inf above) rather than chasing the threshold into the region
    # where the inner solve is ill-conditioned. prof_dev(v̂) = -thr < 0 and, when the
    # profile crosses, prof_dev(cap) > 0 — a clean sign change for the bisection.
    function endpoint(dir)
        cap = v̂ + dir * 8.0
        crossed(prof_dev(cap)) || return nothing      # never crossed in range → boundary
        a, b = v̂, cap
        for _ in 1:24                                  # 16/2²⁴ ≈ 1e-6 precision in log-SD
            m = 0.5 * (a + b)
            crossed(prof_dev(m)) ? (b = m) : (a = m)
        end
        return 0.5 * (a + b)
    end
    lo = endpoint(-1.0); hi = endpoint(+1.0)
    return (sd_lo = lo === nothing ? 0.0 : exp(lo),
            sd_hi = hi === nothing ? Inf : exp(hi))
end

# ---------------------------------------------------------------------------
# Asymmetric fitter (σ-phylo only, mean fixed, 1 free variance param: logL22).
# ---------------------------------------------------------------------------
# Reuses _sigma_re_loadings and _sigma_re_Lambda / _sigma_re_grad convention but
# with an explicit phylogenetic Q instead of Q = I.

# Build Λ = diag(ε², L22²) for the asymmetric case (L22 is the σ-phylo SD).
function _glsp_asym_Λ(logL22::Real)
    L22 = exp(logL22)
    return [_SIGMA_RE_EPS^2 0.0; 0.0 L22^2]
end

# Latent loadings for the asymmetric case: mean axis untouched (Zη = 0),
# σ axis loaded by axis-2 (Zψ = [0 1]).
function _glsp_asym_loadings(n::Int)
    Zη = zeros(n, 2)
    Zψ = zeros(n, 2)
    @views Zψ[:, 2] .= 1.0
    return Zη, Zψ
end

# Marginal NLL at θ = [βμ; βψ; logL22] for the asymmetric σ-phylo case.
function _glsp_asym_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ;
                        warm::Union{Nothing,Ref{Union{Nothing,Vector{Float64}}}} = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    βμ = @view θ[1:pμ]; βψ = @view θ[pμ+1:pμ+pψ]
    logL22 = θ[pμ+pψ+1]
    Λ = _glsp_asym_Λ(logL22)
    Λinv = _ls_inv2x2(Λ)
    P = prior_precision(Q, Λinv)
    a0 = warm === nothing ? nothing : warm[]
    val, a, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ; a0 = a0)
    warm !== nothing && ok && (warm[] = copy(a))
    return ok ? val : 1e18
end

# Exact gradient at θ = [βμ; βψ; logL22].  Embed as [βμ; βψ; log(ε), 0, logL22]
# in the 5-component layout of `_ls_marginal_grad`, then extract only [βμ; βψ; logL22].
function _glsp_asym_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ;
                         warm::Union{Nothing,Ref{Union{Nothing,Vector{Float64}}}} = nothing)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    logL22 = θ[pμ+pψ+1]
    θ_full = vcat(θ[1:pμ+pψ], log(_SIGMA_RE_EPS), 0.0, logL22)
    a0 = warm === nothing ? nothing : warm[]
    g_full = _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ_full, Zη, Zψ; a0 = a0)
    grad = zeros(pμ + pψ + 1)
    grad[1:pμ+pψ] .= g_full[1:pμ+pψ]
    grad[pμ+pψ+1]  = g_full[pμ+pψ+3]   # logL22 component
    return grad
end

# ---------------------------------------------------------------------------
# LBFGS + Newton inner-trust-region outer optimiser (mirrors _fit_locscale).
# `obj` and `grad!` must accept/fill vectors of the appropriate length.
# ---------------------------------------------------------------------------
function _glsp_optimise(obj, grad!, θ0; g_tol = 1e-6, iterations = 1000)
    opts = Optim.Options(g_tol = g_tol, iterations = iterations)
    nm() = Optim.optimize(θ -> obj(θ), θ0, Optim.NelderMead(),
                          Optim.Options(iterations = max(iterations, 2000)))
    res = try
        Optim.optimize(obj, grad!, θ0, Optim.LBFGS(), opts)
    catch err
        err isa InterruptException && rethrow(err)
        try
            nm()
        catch err2
            err2 isa InterruptException && rethrow(err2)
            nm()
        end
    end
    θ̂ = Optim.minimizer(res)
    if any(!isfinite, θ̂) || !(obj(θ̂) < 1e17)
        res = nm()
        θ̂ = Optim.minimizer(res)
    end
    return θ̂, Optim.converged(res)
end

# ---------------------------------------------------------------------------
# Public entry: _fit_gaussian_locscale_phylo
# ---------------------------------------------------------------------------
"""
    _fit_gaussian_locscale_phylo(fam, y, Xμ, Xψ, gidx, G, Q, nmμ, nmσ, grp;
                                  coupled, asymmetric, se, g_tol) -> DrmFit

Fit a UNIVARIATE Gaussian location-scale model with a phylogenetic random effect
on BOTH axes (by default SEPARATE / uncorrelated) or on the σ axis only
(asymmetric).

Modes controlled by kwargs:
  - `coupled = false` (default): SEPARATE block Λ = diag(L11², L22²), L21 ≡ 0.
    Returns DrmFit with `:mu`, `:sigma`, `:resd_mu` (logL11), `:resd_sigma` (logL22).
  - `coupled = true`: FREE L21, full 2×2 Λ with mean↔σ correlation.
    Returns DrmFit with `:mu`, `:sigma`, `:recov` (logL11, logL22, L21).
  - `asymmetric = true`: σ-phylo only (mean fixed effects, no mean phylo RE).
    Returns DrmFit with `:mu`, `:sigma`, `:resd_sigma` (logL22).

The kernel is `Val(:gaussian_mean)`: η = mean, ψ = log σ, integrating the
Gaussian location-scale likelihood through the q=2 augmented-state Laplace spine.

`reml = true` (all three modes) maximises the joint-Laplace restricted likelihood
over (phylo effects, β_μ, β_σ) — native drmTMB's `REML = TRUE` quantity — via
`_glsp_joint_reml_fit`; the reported β are the joint mode at the REML variance
estimates.
"""
function _fit_gaussian_locscale_phylo(fam::Gaussian, y, Xμ, Xψ, gidx, G, Q,
                                       nmμ, nmσ, grp::String;
                                       coupled::Bool = false,
                                       asymmetric::Bool = false,
                                       se::Bool = true,
                                       profile_ci::Bool = false,
                                       reml::Bool = false,
                                       g_tol::Real = 1e-6,
                                       penalty = nothing)
    kind = Val(:gaussian_mean)
    n = length(y)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    # A4c. `drm()` refuses this combination up front; repeat it here because this
    # fitter is also reachable directly (the bridge and the penalty sweep call it).
    (penalty === nothing || !reml) ||
        error("penalty and REML cannot be combined — a penalized fit is a MAP estimator " *
              "and REML is a restricted-likelihood estimator.")

    # ---- ASYMMETRIC: σ-phylo only ----------------------------------------
    if asymmetric
        Zη, Zψ = _glsp_asym_loadings(n)
        # No shared warm (see the separate-block note) — cold inner solves.
        function asym_obj(θ)
            _glsp_asym_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
        end
        function asym_grad!(g, θ)
            g .= _glsp_asym_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
            g
        end
        # A4c penalized-MAP: one σ-phylo SD, at index pμ+pψ+1 on the log scale.
        # `asym_obj` is deliberately left UNPENALIZED so `ml_nll` below stays the
        # data log-likelihood; only the optimiser, the Wald curvature and the
        # profile root-find see the penalized versions.
        _ipen = pμ + pψ + 1
        _asym_grad_raw(θ) = _glsp_asym_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
        pen_obj, pen_gradf = if penalty === nothing
            asym_obj, _asym_grad_raw
        else
            (θ -> asym_obj(θ) + _phylo_pen_apply_single!(nothing, penalty, θ, _ipen)),
            (θ -> (g = _asym_grad_raw(θ); _phylo_pen_apply_single!(g, penalty, θ, _ipen); g))
        end
        # A LOCAL BINDING, not `pen_grad!(g, θ) = ...`: the named form would define
        # three methods of one module-level `pen_grad!` across the three blocks,
        # which is method overwriting and fails precompilation.
        pen_grad! = (g, θ) -> (g .= pen_gradf(θ); g)
        βμ0 = Xμ \ y
        βψ0 = zeros(pψ)
        logL22_0 = log(0.3)
        θ0 = vcat(βμ0, βψ0, logL22_0)
        θ̂, conv = _glsp_optimise(pen_obj, pen_grad!, θ0; g_tol = g_tol)
        ml_nll = asym_obj(θ̂); reml_nll = NaN; V_reml = nothing; ci_reml = nothing
        if reml
            # Native drmTMB's restricted likelihood: ONE joint Laplace over (a, β_μ, β_ψ)
            # with the variance parameter the only outer coordinate (Arc 2; see
            # `_glsp_joint_reml_fit`). Started from a default and from the (boundary-clamped) ML estimate.
            rf = _glsp_joint_reml_fit(kind, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ,
                                      v -> _glsp_asym_Λ(v[1]),
                                      [[log(0.3)], _glsp_reml_start(θ̂[pμ+pψ+1:end], 1:1)],
                                      θ̂[1:pμ+pψ]; se = se, logsd_idx = (1,),
                                      profile_idx = profile_ci ? (1,) : ())
            θ̂ = rf.θ; conv = rf.converged
            ml_nll = rf.ml_nll; reml_nll = rf.reml_nll; V_reml = rf.V; ci_reml = rf.ci
        end
        nll_val = reml ? reml_nll : ml_nll
        βμ̂ = θ̂[1:pμ]; βψ̂ = θ̂[pμ+1:pμ+pψ]; logL22 = θ̂[pμ+pψ+1]
        # Wald covariance via FD of the gradient. Under REML the reported vcov is the
        # inverse Hessian of the RESTRICTED objective (ML info + restricted-penalty
        # curvature, issue #310); under ML it is the ML observed information.
        V = if se
            try
                if reml
                    V_reml   # joint-Laplace REML covariance (β rows: S⁻¹ + J V_vv Jᵀ)
                else
                    # FD of the PENALIZED gradient when a penalty is in force, so the
                    # reported curvature is the MAP curvature (drmTMB says the same:
                    # penalized SEs are credible-interval-shaped, not frequentist).
                    h = 1e-4; np = length(θ̂)
                    H = zeros(np, np)
                    for j in 1:np
                        tp = copy(θ̂); tp[j] += h
                        tm = copy(θ̂); tm[j] -= h
                        gp = pen_gradf(tp)
                        gm = pen_gradf(tm)
                        H[:, j] .= (gp .- gm) ./ (2h)
                    end
                    # PD-guard: at the variance boundary H is singular and `inv` returns GARBAGE
                    # (huge finite values), not an error — so report NaN SEs (use profile_ci there).
                    chH = cholesky(Symmetric(H); check = false)
                    issuccess(chH) ? Matrix(inv(chH)) : fill(NaN, size(H))
                end
            catch
                fill(NaN, length(θ̂), length(θ̂))
            end
        else
            fill(NaN, length(θ̂), length(θ̂))
        end
        blocks = Pair{Symbol,UnitRange{Int}}[
            :mu => 1:pμ,
            :sigma => (pμ+1):(pμ+pψ),
            :resd_sigma => (pμ+pψ+1):(pμ+pψ+1)
        ]
        names = Pair{Symbol,Vector{String}}[
            :mu => nmμ,
            :sigma => nmσ,
            :resd_sigma => ["$(grp):sd_sigma"]
        ]
        means  = Dict(:mu => Xμ * βμ̂)
        obs    = Dict(:mu => Float64.(y))
        scales = Dict(:sigma => exp.(Xψ * βψ̂))
        # B2 — boundary-aware profile CI for the σ-phylo SD (the most boundary-prone
        # cell: Ayumi's collapsing σ-SDs). Reports `[0, x]` honestly when the scale
        # signal is absent. Opt-in (profile_ci) — the root-find re-fits the route NLL.
        if profile_ci
            # Profile off the PENALIZED objective under a penalty, so the interval and
            # the point estimate come from the same surface; under REML, off the
            # restricted surface the point estimate maximises (`_glsp_joint_reml_fit`).
            ci_s = reml ? ci_reml[1] : _glsp_profile_ci(pen_obj, pen_gradf, θ̂, pμ + pψ + 1)
            scales[:profile_ci_sd_sigma] = [ci_s.sd_lo, ci_s.sd_hi]
        end
        fit = DrmFit(fam, blocks, names, θ̂, V, -nll_val, n, conv, means, obs, scales)
        fit = reml ? _withreml(fit, -reml_nll, -ml_nll) : fit
        if penalty !== nothing
            fit = _withmap(fit, _phylo_pen_apply_single!(nothing, penalty, θ̂, _ipen), penalty)
        end
        return fit
    end

    # ---- BOTH-PHYLO: separate (MUST-HAVE) or coupled ----------------------
    Zη = _ls_canonical_Zeta(n)    # [1 0] per row: axis-1 → mean
    Zψ = _ls_canonical_Zpsi(n)    # [0 1] per row: axis-2 → σ

    if !coupled
        # SEPARATE block: 2 free variance params [logL11, logL22]
        # NOTE: NO shared warm-start between obj and grad. The LBFGS line search
        # leaves the grad's warm inner-mode from a DIFFERENT θ (stale), which froze
        # Λ at its init (both SDs stuck at 0.3). Cold inner solves are correct;
        # perf is fine at these sizes.
        function sep_obj(θ)
            _glsp_sep_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
        end
        function sep_grad!(g, θ)
            g .= _glsp_sep_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
            g
        end
        # A4c penalized-MAP: TWO independent phylo SDs at pμ+pψ+1 and pμ+pψ+2. This
        # block constrains the mean↔σ phylo correlation to zero, so there is no
        # correlation parameter here and `cor_sd` is REFUSED (not silently ignored)
        # — see `_phylo_pen_apply_separate!`. `sep_obj` stays unpenalized so
        # `ml_nll` remains the data log-likelihood.
        _isd1 = pμ + pψ + 1; _isd2 = pμ + pψ + 2
        _sep_grad_raw(θ) = _glsp_sep_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
        pen_obj, pen_gradf = if penalty === nothing
            sep_obj, _sep_grad_raw
        else
            (θ -> sep_obj(θ) + _phylo_pen_apply_separate!(nothing, penalty, θ, _isd1, _isd2)),
            (θ -> (g = _sep_grad_raw(θ); _phylo_pen_apply_separate!(g, penalty, θ, _isd1, _isd2); g))
        end
        # A LOCAL BINDING, not `pen_grad!(g, θ) = ...`: the named form would define
        # three methods of one module-level `pen_grad!` across the three blocks,
        # which is method overwriting and fails precompilation.
        pen_grad! = (g, θ) -> (g .= pen_gradf(θ); g)
        βμ0 = Xμ \ y
        βψ0 = zeros(pψ)
        θ0 = vcat(βμ0, βψ0, log(0.3), log(0.3))   # [βμ; βψ; logL11; logL22]
        θ̂, conv = _glsp_optimise(pen_obj, pen_grad!, θ0; g_tol = g_tol)
        ml_nll = sep_obj(θ̂); reml_nll = NaN; V_reml = nothing; ci_reml = nothing
        if reml
            # The same joint-Laplace restricted likelihood as the asymmetric and coupled
            # blocks (Arc 2), with Λ = diag(L11², L22²). No native twin: native drmTMB
            # always estimates the mean↔σ phylo correlation for this formula (the
            # coupled block below is the native-matching route).
            rf = _glsp_joint_reml_fit(kind, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ,
                                      _glsp_sep_Λ,
                                      [[log(0.3), log(0.3)], _glsp_reml_start(θ̂[pμ+pψ+1:end], 1:2)],
                                      θ̂[1:pμ+pψ]; se = se, logsd_idx = (1, 2),
                                      profile_idx = profile_ci ? (2, 1) : ())
            θ̂ = rf.θ; conv = rf.converged
            ml_nll = rf.ml_nll; reml_nll = rf.reml_nll; V_reml = rf.V; ci_reml = rf.ci
        end
        nll_val = reml ? reml_nll : ml_nll
        βμ̂ = θ̂[1:pμ]; βψ̂ = θ̂[pμ+1:pμ+pψ]
        logL11 = θ̂[pμ+pψ+1]; logL22 = θ̂[pμ+pψ+2]
        # Λ for reporting
        Λ̂ = _glsp_sep_Λ([logL11, logL22])
        # Wald covariance via FD of the gradient. Under REML the reported vcov is the
        # inverse Hessian of the RESTRICTED objective (ML info + restricted-penalty
        # curvature, issue #310); under ML it is the ML observed information.
        V = if se
            try
                if reml
                    V_reml   # joint-Laplace REML covariance (β rows: S⁻¹ + J V_vv Jᵀ)
                else
                    h = 1e-4; np = length(θ̂)
                    H = zeros(np, np)
                    for j in 1:np
                        tp = copy(θ̂); tp[j] += h
                        tm = copy(θ̂); tm[j] -= h
                        gp = pen_gradf(tp)
                        gm = pen_gradf(tm)
                        H[:, j] .= (gp .- gm) ./ (2h)
                    end
                    # PD-guard: at the variance boundary H is singular and `inv` returns GARBAGE
                    # (huge finite values), not an error — so report NaN SEs (use profile_ci there).
                    chH = cholesky(Symmetric(H); check = false)
                    issuccess(chH) ? Matrix(inv(chH)) : fill(NaN, size(H))
                end
            catch
                fill(NaN, length(θ̂), length(θ̂))
            end
        else
            fill(NaN, length(θ̂), length(θ̂))
        end
        blocks = Pair{Symbol,UnitRange{Int}}[
            :mu => 1:pμ,
            :sigma => (pμ+1):(pμ+pψ),
            :resd_mu    => (pμ+pψ+1):(pμ+pψ+1),   # logL11
            :resd_sigma => (pμ+pψ+2):(pμ+pψ+2)    # logL22
        ]
        names = Pair{Symbol,Vector{String}}[
            :mu => nmμ,
            :sigma => nmσ,
            :resd_mu    => ["$(grp):sd_mu"],
            :resd_sigma => ["$(grp):sd_sigma"]
        ]
        means  = Dict(:mu => Xμ * βμ̂)
        obs    = Dict(:mu => Float64.(y))
        scales = Dict(:sigma => exp.(Xψ * βψ̂))
        # Attach Lambda and components as extra metadata in the scales dict for
        # downstream accessors (they read :lambda_sd_mu, :lambda_sd_sigma).
        scales[:lambda_sd_mu]    = [sqrt(Λ̂[1, 1])]
        scales[:lambda_sd_sigma] = [sqrt(Λ̂[2, 2])]
        # B2 — boundary-aware profile-likelihood CIs for the phylo SDs (honest
        # `[0, x]` at the boundary, where the Wald V is singular). Opt-in
        # (profile_ci) — the root-find re-optimises the route's own NLL per endpoint.
        if profile_ci
            # Profile the PENALIZED surface under a penalty, so the interval and the
            # point estimate come from the same objective; under REML, the restricted
            # surface (`_glsp_joint_reml_fit`, profile_idx = (σ, μ)).
            ci_s = reml ? ci_reml[1] : _glsp_profile_ci(pen_obj, pen_gradf, θ̂, pμ + pψ + 2)
            ci_m = reml ? ci_reml[2] : _glsp_profile_ci(pen_obj, pen_gradf, θ̂, pμ + pψ + 1)
            scales[:profile_ci_sd_sigma] = [ci_s.sd_lo, ci_s.sd_hi]
            scales[:profile_ci_sd_mu]    = [ci_m.sd_lo, ci_m.sd_hi]
        end
        fit = DrmFit(fam, blocks, names, θ̂, V, -nll_val, n, conv, means, obs, scales)
        fit = reml ? _withreml(fit, -reml_nll, -ml_nll) : fit
        if penalty !== nothing
            fit = _withmap(fit, _phylo_pen_apply_separate!(nothing, penalty, θ̂, _isd1, _isd2), penalty)
        end
        return fit

    else
        # COUPLED block: 3 free variance params [logL11, L21, logL22]. This is the block
        # native drmTMB fits for `mu ~ … + phylo(1 | g)`, `sigma ~ … + phylo(1 | g)`
        # (it estimates the mean↔σ phylo correlation), under ML and — since Arc 2 — REML.
        # No shared warm (see the separate-block note) — cold inner solves.
        function coup_obj(θ)
            pμ_ = size(Xμ, 2); pψ_ = size(Xψ, 2)
            βμ = @view θ[1:pμ_]; βψ = @view θ[pμ_+1:pμ_+pψ_]
            λv = θ[pμ_+pψ_+1:pμ_+pψ_+3]
            Λ  = _glsp_coupled_Λ(λv)
            Λinv = _ls_inv2x2(Λ)
            P = prior_precision(Q, Λinv)
            val, a, ok = _ls_marginal_nll(kind, y, Xμ * βμ, Xψ * βψ, gidx, G, P, Zη, Zψ)
            ok ? val : 1e18
        end
        function coup_grad!(g, θ)
            g .= _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
            g
        end
        # A4c penalized-MAP: the ONLY block with a live phylo correlation. λ =
        # [logL11, L21, logL22] is a Cholesky factor, so the SD penalty applies to
        # log(L11) and log(sqrt(L21²+L22²)), and `cor_sd` — when asked for — applies
        # to atanh(L21/sqrt(L21²+L22²)), which is drmTMB's `eta_cor_phylo`. Penalising
        # L21 itself would be a different prior. `coup_obj` stays unpenalized so
        # `nll_val` below remains the data log-likelihood.
        _i0 = pμ + pψ + 1
        _coup_grad_raw(θ) = _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
        pen_obj, pen_gradf = if penalty === nothing
            coup_obj, _coup_grad_raw
        else
            (θ -> coup_obj(θ) + _phylo_pen_apply_coupled!(nothing, penalty, θ, _i0)),
            (θ -> (g = _coup_grad_raw(θ); _phylo_pen_apply_coupled!(g, penalty, θ, _i0); g))
        end
        # A LOCAL BINDING, not `pen_grad!(g, θ) = ...`: the named form would define
        # three methods of one module-level `pen_grad!` across the three blocks,
        # which is method overwriting and fails precompilation.
        pen_grad! = (g, θ) -> (g .= pen_gradf(θ); g)
        βμ0 = Xμ \ y
        βψ0 = zeros(pψ)
        θ0 = vcat(βμ0, βψ0, log(0.3), 0.0, log(0.3))
        θ̂, conv = _glsp_optimise(pen_obj, pen_grad!, θ0; g_tol = g_tol)
        nll_val = coup_obj(θ̂); ml_nll = nll_val; reml_nll = NaN; V_reml = nothing
        if reml
            # Joint-Laplace REML over (a, β_μ, β_ψ) — native drmTMB's restricted likelihood
            # for this shape (Arc 2; see `_glsp_joint_reml_fit`), including its correlation bound.
            rf = _glsp_joint_reml_fit(kind, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ,
                                      _glsp_coupled_Λ,
                                      [[log(0.3), 0.0, log(0.3)],
                                       _glsp_reml_start(θ̂[pμ+pψ+1:end], (1, 3))],
                                      θ̂[1:pμ+pψ]; se = se, logsd_idx = (1, 3),
                                      shrink_idx = (2,), cor_edge = true)
            θ̂ = rf.θ; conv = rf.converged
            ml_nll = rf.ml_nll; reml_nll = rf.reml_nll; nll_val = reml_nll; V_reml = rf.V
        end
        βμ̂ = θ̂[1:pμ]; βψ̂ = θ̂[pμ+1:pμ+pψ]
        λ̂ = θ̂[pμ+pψ+1:pμ+pψ+3]
        Λ̂ = _glsp_coupled_Λ(λ̂)
        comp = _ls_components(Λ̂)
        # Wald via FD of the general gradient (ML); the joint-Laplace covariance under REML.
        V = if reml
            V_reml
        elseif se
            try
                h = 1e-4; np = length(θ̂)
                H = zeros(np, np)
                for j in 1:np
                    tp = copy(θ̂); tp[j] += h
                    tm = copy(θ̂); tm[j] -= h
                    gp = pen_gradf(tp)
                    gm = pen_gradf(tm)
                    H[:, j] .= (gp .- gm) ./ (2h)
                end
                # PD-guard: at the variance boundary H is singular and `inv` returns GARBAGE
                # (huge finite values), not an error — so report NaN SEs (use profile_ci there).
                # Under ML the ML observed information is the correct Wald curvature.
                chH = cholesky(Symmetric(H); check = false)
                issuccess(chH) ? Matrix(inv(chH)) : fill(NaN, size(H))
            catch
                fill(NaN, length(θ̂), length(θ̂))
            end
        else
            fill(NaN, length(θ̂), length(θ̂))
        end
        # `:recov` block: [logL11, logL22, L21] — matches the locscale_frontend convention.
        theta_out = vcat(βμ̂, βψ̂, λ̂[1], λ̂[3], λ̂[2])
        perm = vcat(collect(1:(pμ+pψ)), [pμ+pψ+1, pμ+pψ+3, pμ+pψ+2])
        V_out = V === nothing ? fill(NaN, length(theta_out), length(theta_out)) :
                (all(isnan, V) ? fill(NaN, length(theta_out), length(theta_out)) : V[perm, perm])
        blocks = Pair{Symbol,UnitRange{Int}}[
            :mu    => 1:pμ,
            :sigma => (pμ+1):(pμ+pψ),
            :recov => (pμ+pψ+1):(pμ+pψ+3)
        ]
        names = Pair{Symbol,Vector{String}}[
            :mu    => nmμ,
            :sigma => nmσ,
            :recov => ["$(grp):L11", "$(grp):L22", "$(grp):L21"]
        ]
        means  = Dict(:mu => Xμ * βμ̂)
        obs    = Dict(:mu => Float64.(y))
        scales = Dict(:sigma => exp.(Xψ * βψ̂))
        scales[:lambda_sd_mu]    = [comp.sd_mu]
        scales[:lambda_sd_sigma] = [comp.sd_psi]
        scales[:lambda_cor]      = [comp.cor_mu_psi]
        fit = DrmFit(fam, blocks, names, theta_out, V_out, -nll_val, n, conv, means, obs, scales)
        fit = reml ? _withreml(fit, -reml_nll, -ml_nll) : fit
        if penalty !== nothing
            fit = _withmap(fit, _phylo_pen_apply_coupled!(nothing, penalty, θ̂, _i0), penalty)
        end
        return fit
    end
end

# ---------------------------------------------------------------------------
# Convenience accessor: extract the two phylo SDs from a separate-block fit.
# Returns NamedTuple (sd_mu, sd_sigma).
# ---------------------------------------------------------------------------
"""
    gaussian_locscale_phylo_sds(fit::DrmFit) -> NamedTuple

Extract sd(μ-phylo) and sd(σ-phylo) from a SEPARATE-block Gaussian phylo
location-scale fit. Reads the `:resd_mu` / `:resd_sigma` blocks (exp of their
stored log values). For a COUPLED fit use `fit.scales[:lambda_sd_mu]` etc.
"""
function gaussian_locscale_phylo_sds(fit::DrmFit)
    has_sep_mu  = any(p -> p === :resd_mu,    first.(fit.blocks))
    has_sep_sig = any(p -> p === :resd_sigma, first.(fit.blocks))
    if has_sep_mu && has_sep_sig
        return (sd_mu = exp(coef(fit, :resd_mu)[1]),
                sd_sigma = exp(coef(fit, :resd_sigma)[1]))
    elseif has_sep_sig && !has_sep_mu
        return (sd_mu = 0.0,
                sd_sigma = exp(coef(fit, :resd_sigma)[1]))
    elseif haskey(fit.scales, :lambda_sd_mu) && haskey(fit.scales, :lambda_sd_sigma)
        return (sd_mu = fit.scales[:lambda_sd_mu][1],
                sd_sigma = fit.scales[:lambda_sd_sigma][1])
    else
        error("fit does not appear to be a gaussian_locscale_phylo fit")
    end
end
