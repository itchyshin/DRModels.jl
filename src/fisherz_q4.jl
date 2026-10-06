# fisherz_q4.jl — q=4 spherical/LKJ (D·R·D) OUTER reparameterization of the 4×4
# among-axis covariance Σ_a, wrapping the UNTOUCHED sparse exact-gradient engine
# (marginal_and_exact_grad, fit_q4_sparse_tmb.jl).
#
# ENGINE SANCTITY: this file does NOT modify the O(p) inner gradient. It is a
# pure OUTER reparameterization. The engine's native parameter for Σ_a is the
# 10-vector log-Cholesky `lc` (lc_to_Λ / Λ_to_lc, fit_ml_q4.jl). Here we replace
# `lc` with a separation strategy (Barnard et al. 2000):
#
#     Σ_a = D · R · D,   D = diag(exp.(d)),   R = C Cᵀ  (correlation matrix)
#
# φ_a = [d (4 log-SDs); θ_R (6 reals)]  (10 dims, same count as lc).
#   • D = diag of the 4 axis standard deviations (exp keeps them > 0).
#   • R is built from a lower-triangular spherical/LKJ correlation-Cholesky C:
#     row i is a UNIT vector in ℝ^i parameterized by (i-1) angles in (0,π) via
#     tanh → angle, so R = C Cᵀ is ALWAYS PD with an EXACT unit diagonal over
#     ALL of ℝ⁶. NOTE: this is an LKJ/spherical angle map, NOT the Fisher-z link.
#     For the 2×2 case the single angle is α = π·(tanh θ + 1)/2 and C[2,1]=cos α,
#     so ρ = R[2,1] = cos(π·(tanh θ + 1)/2) — distinct from the q2 Fisher-z
#     bijection ρ = tanh(θ). Only the PD/interior property carries over from the
#     verified q2 path; the specific link (and hence the θ→ρ back-transform)
#     differs, so a fitted θ_R must be inverted with ρ = cos(π·(tanh θ_R + 1)/2).
#
# This is the 4×4 spherical/LKJ analogue of the q2 correlation reparameterisation
# (verified in ML and REML, /tmp/drm-validation/fisherz_*). Its value is
# CONDITIONING / ROBUSTNESS at
# the σ-collapse boundary (one axis SD → 0), NOT manufacturing signal: the native
# log-Cholesky lets a correlation pin at ±1 with an NA/ill-conditioned Hessian as
# an axis variance collapses; the separation form keeps R interior and PD.
#
# PER OUTER EVAL (φ_a → engine):
#   1. lc = Λ_to_lc(DRD(φ_a))                 (fit_ml_q4.jl:16)
#   2. call the UNTOUCHED marginal_and_exact_grad → g_lc (10 phylocov) + β-grad
#   3. chain-rule g_φ = Jᵀ g_lc,  J = ForwardDiff.jacobian(φ -> Λ_to_lc(DRD(φ)), φ)
#      The β-block (mean/scale/rho fixed effects) passes straight through.
#
# Σ_a (= D R D) and the 6 among-axis correlations are reported directly.

using LinearAlgebra, ForwardDiff, Optim

# ---------------------------------------------------------------------------
# Spherical / LKJ correlation-Cholesky over the 6 reals θ_R.
# Lower-tri C (4×4): row i has unit norm, built from (i-1) angles. The angle for
# the (i, j<i) slot is φ = π · (tanh(θ)+1)/2 ∈ (0,π). Standard construction:
#   C[i,1]   = cos(α_{i,1})
#   C[i,k]   = cos(α_{i,k}) · Π_{l<k} sin(α_{i,l})   (1 < k < i)
#   C[i,i]   = Π_{l<i} sin(α_{i,l})                   (the remaining length ⇒ ‖row‖=1)
# θ_R order matches the strictly-lower-triangular column-major sweep used by lc's
# off-diagonals: (2,1),(3,1),(4,1),(3,2),(4,2),(4,3).
# ---------------------------------------------------------------------------
"Lower-tri correlation-Cholesky C (4×4) from the 6 angle-reals θ_R (R = C Cᵀ)."
function fz_R_chol(θR::AbstractVector{T}) where {T}
    C = zeros(T, 4, 4)
    C[1, 1] = one(T)
    # angle index in θR (strictly-lower-tri, column-major): map (i,j) -> position.
    #   (2,1)->1 (3,1)->2 (4,1)->3 (3,2)->4 (4,2)->5 (4,3)->6
    angidx = Dict((2, 1) => 1, (3, 1) => 2, (4, 1) => 3,
                  (3, 2) => 4, (4, 2) => 5, (4, 3) => 6)
    for i in 2:4
        sin_prod = one(T)
        for k in 1:(i - 1)
            α = T(π) * (tanh(θR[angidx[(i, k)]]) + one(T)) / 2   # ∈ (0,π)
            C[i, k] = cos(α) * sin_prod
            sin_prod *= sin(α)
        end
        C[i, i] = sin_prod                                       # makes ‖row i‖ = 1
    end
    return C
end

"4×4 correlation matrix R = C Cᵀ from the 6 angle-reals θ_R (always PD, unit diag)."
fz_R(θR::AbstractVector) = (C = fz_R_chol(θR); C * C')

# ---------------------------------------------------------------------------
# Σ_a = D R D from φ_a = [d(4); θ_R(6)].  D = diag(exp.(d)).
# ---------------------------------------------------------------------------
"Σ_a = D R D (4×4 PD) from φ_a = [d (4 log-SDs); θ_R (6 angle-reals)]."
function fz_DRD(φa::AbstractVector{T}) where {T}
    d  = @view φa[1:4]
    θR = @view φa[5:10]
    sd = exp.(d)
    R  = fz_R(θR)
    return (sd .* R) .* sd'        # D R D  (broadcast diag scaling, AD-friendly)
end

# Type-generic log-Cholesky (4×4 → 10-vec). MIRRORS the engine's Λ_to_lc
# (fit_ml_q4.jl:16) EXACTLY — same column-major lower-tri sweep, log on the
# diagonal — but builds an eltype-following vector so ForwardDiff.Dual flows for
# the J = ∂lc/∂φ_a chain rule. The engine's own Λ_to_lc (Float64-only `push!`)
# is left UNTOUCHED; this is a private AD-safe twin used only inside the wrapper.
function _fz_Λ_to_lc(Λ::AbstractMatrix{T}) where {T}
    C = cholesky(Symmetric(Λ)).L
    v = Vector{T}(undef, 10)
    k = 0
    for j in 1:4, i in j:4
        k += 1
        v[k] = i == j ? log(C[i, j]) : C[i, j]
    end
    return v
end

# φ_a → engine 10-vector lc (so the UNTOUCHED engine consumes it natively).
fz_phi_to_lc(φa::AbstractVector) = _fz_Λ_to_lc(fz_DRD(φa))

# ---------------------------------------------------------------------------
# φ_a → 6 among-axis correlations (upper-tri of R), in the same column-major
# order as θ_R: (2,1),(3,1),(4,1),(3,2),(4,2),(4,3).
# Axes are (mu1, mu2, sigma1, sigma2): so e.g. (3,2)=cor(sigma1,mu2),
# (4,3)=cor(sigma2,sigma1).
# ---------------------------------------------------------------------------
function fz_correlations(φa::AbstractVector)
    Σ = fz_DRD(φa)
    s = sqrt.(diag(Σ))
    cor(i, j) = Σ[i, j] / (s[i] * s[j])
    return (cor(2, 1), cor(3, 1), cor(4, 1), cor(3, 2), cor(4, 2), cor(4, 3))
end

# Initial φ_a from a starting Σ_a (e.g. the engine's Λ0): recover SDs and angles
# by inverting the spherical construction on the correlation Cholesky of R.
"Initial φ_a = [d; θ_R] reproducing a starting Σ_a (used to seed the optimiser)."
function fz_init_from_Sigma(Σ0::AbstractMatrix)
    # Floor the axis SDs so a collapsed variance (diag(Σ0) ≈ 0) does not give
    # d = log(0) = -Inf or 0/0 = NaN correlation entries. 1e-8 on the variance
    # keeps d finite while staying negligible relative to any estimable axis.
    s = sqrt.(max.(diag(Σ0), 1e-8))
    d = log.(s)
    R = Symmetric((Σ0 ./ s) ./ s')           # correlation matrix
    # A near-singular but PD correlation matrix can still throw PosDefException;
    # mirror _q4_re_prior_chol's ascending ridge so the corr-Cholesky is robust.
    local C
    C = nothing
    for ridge in (0.0, 1e-10, 1e-8, 1e-6)
        Rr = ridge == 0.0 ? R : Symmetric(Matrix(R) + ridge * I)
        F = cholesky(Rr; check = false)
        if issuccess(F)
            C = F.L                           # lower-tri Cholesky (unit rows ⇒ corr-chol)
            break
        end
    end
    C === nothing && error("fz_init_from_Sigma: could not factor the correlation " *
                           "matrix of Σ0 (an axis is fully collapsed); try fewer axes")
    θR = zeros(6)
    angidx = Dict((2, 1) => 1, (3, 1) => 2, (4, 1) => 3,
                  (3, 2) => 4, (4, 2) => 5, (4, 3) => 6)
    for i in 2:4
        sin_prod = 1.0
        for k in 1:(i - 1)
            c = clamp(C[i, k] / sin_prod, -1.0, 1.0)
            α = acos(c)                        # ∈ (0,π)
            # invert α = π(tanh θ + 1)/2  ⇒  θ = atanh(2α/π − 1)
            θR[angidx[(i, k)]] = atanh(clamp(2α / π - 1.0, -0.999999, 0.999999))
            sin_prod *= sin(α)
        end
    end
    return vcat(d, θR)
end

# ---------------------------------------------------------------------------
# D·R·D OUTER objective + transformed gradient, wrapping the engine.
# Outer parameter vector ψ = [β (k1+k2+ks1+ks2+kr); φ_a (10)]. The β-block maps
# 1:1 to the engine's β; φ_a maps to lc. The transformed gradient:
#   g_β = g_β(engine);  g_φ = Jᵀ g_lc(engine),  J = ∂lc/∂φ_a (10×10).
# ---------------------------------------------------------------------------
"Length of ψ = nβ + 10 (the D·R·D outer parameter vector)."
fz_psi_len(prob::AugProblem) = theta_len(prob)   # same total dimension as θ

"Split ψ into the β-block (Float-passthrough) and φ_a (10)."
function fz_unpack_psi(prob::AugProblem, ψ::AbstractVector)
    nβ = theta_len(prob) - 10
    return (@view ψ[1:nβ]), (@view ψ[nβ+1:nβ+10])
end

"ψ (D·R·D outer) → engine θ (β + lc). Pure reparameterization of the Σ_a block."
function fz_psi_to_theta(prob::AugProblem, ψ::AbstractVector)
    βblk, φa = fz_unpack_psi(prob, ψ)
    return vcat(Vector{Float64}(βblk), fz_phi_to_lc(Vector{Float64}(φa)))
end

"""
    fz_marginal_and_grad(prob, Q_cond, ψ; u0, n_newton) -> (nll, g_ψ, û, chH)

TRUE sparse Laplace NLL and its EXACT gradient in the D·R·D OUTER parameters
ψ = [β; φ_a]. Calls the UNTOUCHED `marginal_and_exact_grad` on θ = [β; lc(φ_a)],
then chain-rules the Σ_a block: g_φ = Jᵀ g_lc with J = ∂lc/∂φ_a. β passes through.
"""
function fz_marginal_and_grad(prob::AugProblem, Q_cond::SparseMatrixCSC,
                              ψ::AbstractVector{Float64};
                              u0 = nothing, n_newton::Int = 40)
    nβ = theta_len(prob) - 10
    βblk = ψ[1:nβ]
    φa   = ψ[nβ+1:nβ+10]
    θ    = vcat(βblk, fz_phi_to_lc(φa))

    # UNTOUCHED engine: exact O(p) NLL + 17-dim (here nβ+10) gradient in θ-space.
    nll, g_θ, û, chH = marginal_and_exact_grad(prob, Q_cond, θ; u0 = u0, n_newton = n_newton)

    g_ψ = similar(ψ)
    g_ψ[1:nβ] .= @view g_θ[1:nβ]                       # β-block passes straight through
    J = ForwardDiff.jacobian(fz_phi_to_lc, φa)         # 10×10 = ∂lc/∂φ_a
    g_ψ[nβ+1:nβ+10] .= J' * (@view g_θ[nβ+1:nβ+10])    # chain rule on the Σ_a block
    return nll, g_ψ, û, chH
end

"D·R·D outer marginal NLL only (for FD verification / line search)."
function fz_marginal_nll(prob::AugProblem, Q_cond::SparseMatrixCSC,
                         ψ::AbstractVector{Float64}; u0 = nothing, n_newton::Int = 40)
    θ = fz_psi_to_theta(prob, ψ)
    return marginal_nll(prob, Q_cond, θ; u0 = u0, n_newton = n_newton)[1]
end

# ---------------------------------------------------------------------------
# Fit driver: same fg!/LBFGS pattern as fit_q4_sparse_tmb, but in ψ-space.
# Hard-cap defaults match ENGINE SANCTITY (callers may tighten further).
# ---------------------------------------------------------------------------
"""
    fit_q4_sparse_fisherz(prob, Q_cond; β0, Σa0 / Λ0, ...) -> NamedTuple

Fit the q=4 PLSM in the separation (`D·R·D`) OUTER parameterization, with
`Σ_a = D R D`, `D = diag(exp.(d))` and `R = C Cᵀ` a correlation matrix. The inner
engine is the UNTOUCHED `marginal_and_exact_grad`. Returns `Σ_a` and the 6
among-axis correlations directly, alongside the usual fit diagnostics.

!!! warning "The name says Fisher-z; the map is not Fisher-z"
    `fisherz` here, and the `fz_` prefix throughout, is historical. `R` is built
    from a spherical/LKJ correlation-Cholesky whose angles are
    `α = π·(tanh θ + 1)/2 ∈ (0, π)`, so a fitted angle-real `θ` back-transforms as
    `ρ = cos(π·(tanh θ + 1)/2)` — **not** the Fisher-z bijection `ρ = tanh θ`.
    Inverting with `tanh` returns the wrong correlation. Only the PD/interior
    property carries over from the verified q=2 Fisher-z path, not the link
    itself. The 6 among-axis correlations are returned already on the `ρ` scale,
    so prefer them to inverting `θ_R` by hand.
"""
function fit_q4_sparse_fisherz(prob::AugProblem, Q_cond::SparseMatrixCSC;
                               β0 = nothing, Λ0 = nothing, Σa0 = nothing,
                               ψ0 = nothing,
                               g_tol::Float64 = 1e-3, iterations::Int = 200,
                               n_newton::Int = 40, show_trace::Bool = false,
                               time_limit::Float64 = 25.0,
                               linesearch = Optim.LineSearches.MoreThuente())
    if ψ0 === nothing
        β0 === nothing && error("supply ψ0 or (β0, Λ0/Σa0)")
        Σstart = Σa0 !== nothing ? Matrix(Σa0) :
                 Λ0  !== nothing ? Matrix(Λ0)  : Matrix(0.3 * I(4))
        φa0 = fz_init_from_Sigma(Σstart)
        βvec0 = vcat(β0.mu1, β0.mu2, β0.s1, β0.s2, β0.rho)
        ψ0 = vcat(βvec0, φa0)
    end
    ψ0 = Vector{Float64}(ψ0)

    u_cache = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    nobs = length(prob.leaf_node)

    fg! = function (F, G, ψ)
        local nll, g, û
        try
            nll, g, û, _ = fz_marginal_and_grad(prob, Q_cond, Vector{Float64}(ψ);
                                                u0 = u_cache[], n_newton = n_newton)
        catch e
            # Implicit-constraint barrier: a trial φ_a from the line search can
            # push a log-SD/angle so far that DRD(φ) overflows to Inf/NaN and the
            # Cholesky in _fz_Λ_to_lc throws ArgumentError, or the Laplace mode
            # loses PD-ness. Return Inf (no gradient) so the line search shrinks
            # the step — same robustness barrier as the native fit_q4_sparse_tmb.
            (e isa DomainError || e isa LinearAlgebra.PosDefException ||
             e isa LinearAlgebra.SingularException || e isa ArgumentError) || rethrow(e)
            return Inf
        end
        any(!isfinite, g) && return Inf
        u_cache[] = û
        G !== nothing && copyto!(G, g ./ nobs)     # mean (per-obs) objective, scale-invariant
        return nll / nobs
    end
    od = Optim.NLSolversBase.only_fg!(fg!)

    res = Optim.optimize(
        od, ψ0,
        LBFGS(alphaguess = Optim.LineSearches.InitialStatic(scaled = true),
              linesearch = linesearch),
        Optim.Options(g_tol = g_tol, f_reltol = 1e-7, successive_f_tol = 3,
                      iterations = iterations, time_limit = time_limit,
                      show_trace = show_trace, show_every = 1),
    )

    ψ_hat = Optim.minimizer(res)
    βblk, φa_hat = fz_unpack_psi(prob, ψ_hat)
    k1, k2, ks1, ks2, kr = beta_widths(prob)
    o1 = 0; o2 = k1; o3 = o2 + k2; o4 = o3 + ks1; o5 = o4 + ks2
    β_hat = (mu1 = collect(βblk[o1+1:o1+k1]), mu2 = collect(βblk[o2+1:o2+k2]),
             s1 = collect(βblk[o3+1:o3+ks1]), s2 = collect(βblk[o4+1:o4+ks2]),
             rho = collect(βblk[o5+1:o5+kr]))
    Σa_hat = fz_DRD(Vector{Float64}(φa_hat))
    cors = fz_correlations(Vector{Float64}(φa_hat))
    nll_mean = _objective_at_minimizer_fg(fg!, res)   # NOT Optim.minimum(res): see optim_minimum_guard.jl
    return (
        ψ = ψ_hat,
        φa = Vector{Float64}(φa_hat),
        β = β_hat,
        Σa = Σa_hat,
        Λ = Σa_hat,                              # alias: same object the ML path calls Λ
        correlations = cors,                     # (c21,c31,c41,c32,c42,c43) over axes (μ1,μ2,σ1,σ2)
        sds = sqrt.(diag(Σa_hat)),
        nll = nll_mean * nobs,
        loglik = -nll_mean * nobs,
        converged = Optim.converged(res),
        iterations = Optim.iterations(res),
        g_residual = Optim.g_residual(res),
        f_calls = Optim.f_calls(res),
        g_calls = Optim.g_calls(res),
    )
end
