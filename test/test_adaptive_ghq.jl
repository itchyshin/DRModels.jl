# #834: per-group adaptive Gauss–Hermite quadrature for correlated random slopes
# (1 + x | g) on non-Gaussian families.
#
# Before #834 every `_fit_*_corr_ranef` route integrated each group's (b0, b1) on a
# fixed 12×12 grid on the PRIOR scale (b = √2 L z). With informative groups the
# posterior is a narrow spike and the grid misses it: on the Poisson DGP below the
# reported logLik was 34 nat off and the RE correlation came out −0.56 (true MLE
# +0.43). The reference here is an INDEPENDENT AGHQ-40 implementation (ForwardDiff
# Newton to each group's mode, 40×40 nodes), not the package helper.
#
# Test stability fix (2026-09-27): the DGP originally drew data from Julia's global
# RNG (`Random.seed!(n)` + unqualified `rand`/`randn`). That stream is NOT stable
# across Julia releases -- confirmed by running this file on Julia 1.10 and 1.13:
# same seed, different y/s realizations, so the golden logLik/ρ/coef thresholds
# below (tuned to one specific draw) failed on 1.13 while passing on 1.10. Fixed by
# threading an explicit `StableRNG` (already a test dep, used the same way in
# test_mixed_family.jl / test_cox_reid_poisson_phylo.jl) through the whole DGP, and
# re-deriving the golden numbers from the resulting (now version-stable) dataset;
# verified bit-identical fit results on Julia 1.10 and 1.13.
using DRModels
using Test, Random, LinearAlgebra, ForwardDiff, StableRNGs
import Distributions
const _Dd834 = Distributions
const _Optim834 = DRModels.Optim

# --- independent reference ----------------------------------------------------
function _gh834(K)
    β = [sqrt(k / 2) for k in 1:(K-1)]
    E = eigen(SymTridiagonal(zeros(K), β))
    return E.values, sqrt(π) .* E.vectors[1, :] .^ 2
end

# log ∫ exp(h(b)) db for one group: mode by damped Newton, then K×K AGHQ
# (K = 1 is returned as the textbook Laplace formula, computed separately).
function _ref_group834(h; K = 40)
    b = zeros(2)
    for _ in 1:200
        gr = ForwardDiff.gradient(h, b); Hm = -ForwardDiff.hessian(h, b)
        δ = Hm \ gr; t = 1.0
        while h(b + t * δ) < h(b) - 1e-12 && t > 1e-8
            t /= 2
        end
        b += t * δ
        norm(t * δ) < 1e-12 && break
    end
    Hm = -ForwardDiff.hessian(h, b)
    K == 1 && return h(b) + log(2π) - 0.5 * logdet(Hm)       # Laplace
    C = cholesky(Symmetric(inv(Hm))).L
    z, w = _gh834(K); lw = log.(w)
    terms = [lw[j] + lw[k] + h(b + sqrt(2) * C * [z[j], z[k]]) + z[j]^2 + z[k]^2 for j in 1:K for k in 1:K]
    mx = maximum(terms)
    return log(2) + logdet(C) + mx + log(sum(exp.(terms .- mx)))
end

function _ref_loglik834(llf, g, η0, x, θre; K = 40)
    L = [exp(θre[1]) 0; θre[3] exp(θre[2])]; S = L * L'; Si = inv(S); ldS = logdet(S)
    tot = 0.0
    for j in 1:maximum(g)
        idx = findall(==(j), g)
        h(b) = sum(llf(i, η0[i] + b[1] + b[2] * x[i]) for i in idx) - 0.5 * dot(b, Si * b) - log(2π) - 0.5 * ldS
        tot += _ref_group834(h; K)
    end
    return tot
end

_rho834(θre) = (L = [exp(θre[1]) 0; θre[3] exp(θre[2])]; S = L * L'; S[1, 2] / sqrt(S[1, 1] * S[2, 2]))

# Shared DGP: G groups × m obs, Σ = [0.5², ρ·0.5·0.4; ·, 0.4²], ρ = 0.3.
# Uses StableRNG (not the global RNG via Random.seed!) so the drawn data — and every
# numeric threshold below that depends on it — is identical across Julia versions;
# Julia's default global RNG stream for a given `Random.seed!(n)` is NOT guaranteed
# stable across Julia releases (observed: Julia 1.10 vs 1.13 diverge). The returned
# `rng` continues to drive the response draws (y / s) so the whole DGP is one stream.
function _dgp834(seed; G = 150, m = 20)
    rng = StableRNG(seed)
    n = G * m; g = repeat(1:G, inner = m); x = randn(rng, n)
    Σ = [0.25 0.3*0.5*0.4; 0.3*0.5*0.4 0.16]
    B = cholesky(Symmetric(Σ)).L * randn(rng, 2, G)
    return n, g, x, B, rng
end

@testset "adaptive GHQ helper (#834)" begin
    rng834 = StableRNG(1)   # version-stable: see _dgp834
    m = 20; x = randn(rng834, m); y = Float64.(rand(rng834, 0:6, m))
    Zre = hcat(ones(m), x); idx = collect(1:m)
    θ = [0.3, 0.2, log(0.5), log(0.4), 0.1]
    function helper(θ, K)
        η0 = θ[1] .+ θ[2] .* x
        L = DRModels._corr_ranef_L(θ[3], θ[4], θ[5])
        ll = (i, η) -> y[i] * η - exp(η)
        return DRModels._aghq_group_logint(ll, idx, η0, Zre, L, DRModels._AGHQRule(2, K), zeros(2))[1]
    end
    L = DRModels._corr_ranef_L(θ[3], θ[4], θ[5]); S = L * L'; Si = inv(S)
    h(b) = sum(y[i] * (θ[1] + θ[2] * x[i] + b[1] + b[2] * x[i]) - exp(θ[1] + θ[2] * x[i] + b[1] + b[2] * x[i]) for i in 1:m) -
           0.5 * dot(b, Si * b) - log(2π) - 0.5 * logdet(S)
    ref40 = _ref_group834(h; K = 40)
    # brute-force 2-D Riemann sum on a fine grid (independent of any quadrature rule)
    hstep = 0.01; grid = -4:hstep:4
    vals = [h([b0, b1]) for b0 in grid for b1 in grid]; mx = maximum(vals)
    brute = mx + log(sum(exp.(vals .- mx)) * hstep^2)
    @test ref40 ≈ brute atol = 1e-6
    @test helper(θ, 40) ≈ ref40 atol = 1e-8                # AGHQ-40 vs independent AGHQ-40
    @test helper(θ, 1) ≈ _ref_group834(h; K = 1) atol = 1e-9   # K = 1 is exactly Laplace
    @test abs(helper(θ, DRModels._CORR_RANEF_AGHQ_K) - ref40) < 1e-3
    @test abs(helper(θ, 1) - ref40) > 1e-3                  # Laplace is not AGHQ (non-vacuous)
    # ForwardDiff gradient and Hessian through the inner Newton match finite differences
    for K in (1, DRModels._CORR_RANEF_AGHQ_K)
        gad = ForwardDiff.gradient(t -> helper(t, K), θ)
        e(j) = Float64.(1:5 .== j)
        gfd = [(helper(θ .+ 1e-6 .* e(j), K) - helper(θ .- 1e-6 .* e(j), K)) / 2e-6 for j in 1:5]
        @test gad ≈ gfd rtol = 1e-6
        Had = ForwardDiff.hessian(t -> helper(t, K), θ)
        Hfd = reduce(hcat, [(ForwardDiff.gradient(t -> helper(t, K), θ .+ 1e-5 .* e(j)) -
                             ForwardDiff.gradient(t -> helper(t, K), θ .- 1e-5 .* e(j))) / 2e-5 for j in 1:5])
        @test Had ≈ Hfd rtol = 1e-5
    end
    # q-generic rule: K^q nodes, weights integrate exp(-zᵀz) exactly
    r3 = DRModels._AGHQRule(3, 4)
    @test size(r3.Z) == (3, 64)
    @test sum(exp.(r3.lw .- vec(sum(abs2, r3.Z; dims = 1)))) ≈ π^(3 / 2) rtol = 1e-12
end

@testset "Poisson (1 + x | g): AGHQ logLik, correlation, start-independence (#834)" begin
    n, g, x, B, rng834 = _dgp834(20260627)
    y = Float64.([rand(rng834, _Dd834.Poisson(exp(1.0 + 0.5 * x[i] + B[1, g[i]] + B[2, g[i]] * x[i]))) for i in 1:n])
    fit = drm(bf(@formula(y ~ x + (1 + x | g))), Poisson(); data = (; y, x, g))
    θ̂ = fit.theta
    lf = [_Dd834.logfactorial(Int(v)) for v in y]
    ref = _ref_loglik834((i, η) -> y[i] * η - exp(η) - lf[i], g, θ̂[1] .+ θ̂[2] .* x, x, θ̂[3:5])
    @test fit.converged
    @test abs(loglik(fit) - ref) < 0.05
    @test ref > -6148.8                                     # true-logLik maximum ≈ −6148.750 (AGHQ-40)
    @test _rho834(θ̂[3:5]) > 0                              # true ρ = 0.3, sign must be recovered
    @test abs(_rho834(θ̂[3:5]) - 0.4012) < 0.02             # AGHQ-15/40 MLE ρ = 0.4012
    @test coef(fit, :mu) ≈ [1.0761, 0.5062] atol = 2e-3     # AGHQ MLE
    # start-independence: the stored objective from two far-apart starts
    θA = [log(sum(y) / n), 0.0, log(0.4), log(0.4), 0.0]
    θB = [0.0, 0.3, log(1.0), log(0.2), 0.3]
    rA = _Optim834.optimize(fit.nll, θA, _Optim834.LBFGS(), _Optim834.Options(g_tol = 1e-8); autodiff = :forward)
    rB = _Optim834.optimize(fit.nll, θB, _Optim834.LBFGS(), _Optim834.Options(g_tol = 1e-8); autodiff = :forward)
    @test _Optim834.minimizer(rA)[1:2] ≈ _Optim834.minimizer(rB)[1:2] atol = 1e-3
    @test _Optim834.minimum(rA) ≈ _Optim834.minimum(rB) atol = 1e-4
end

@testset "Beta-binomial (1 + x | g): AGHQ logLik and correlation (#834)" begin
    n, g, x, B, rng834 = _dgp834(20260628)
    φ = 30.0
    μ = 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x .+ B[1, g] .+ B[2, g] .* x)))
    s = Float64.([rand(rng834, _Dd834.BetaBinomial(20, μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n])
    fail = 20 .- s
    fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 + x | g)), @formula(sigma ~ 1)), BetaBinomial(); data = (; s, fail, x, g))
    θ̂ = fit.theta; φ̂ = exp(-2θ̂[3])
    llf = (i, η) -> (p = 1 / (1 + exp(-η)); _Dd834.logpdf(_Dd834.BetaBinomial(20, p * φ̂, (1 - p) * φ̂), Int(s[i])))
    ref = _ref_loglik834(llf, g, θ̂[1] .+ θ̂[2] .* x, x, θ̂[4:6])
    @test fit.converged
    @test abs(loglik(fit) - ref) < 0.05
    @test ref > -7525.72                                    # true-logLik maximum ≈ −7525.668 (AGHQ-40)
    @test abs(_rho834(θ̂[4:6]) - 0.3333) < 0.02             # AGHQ-15/40 MLE ρ = 0.3333
    @test coef(fit, :mu) ≈ [0.1041, 0.5046] atol = 2e-3     # AGHQ MLE
end

# Uncentred slope covariate (x ≈ 27) with ρ = 0.95: intercept and slope effects are
# nearly collinear in each group's design, the regime where 2×2 determinants computed
# as m11·m22 − m21² cancel catastrophically (#837). The helper takes every
# log-determinant from Cholesky diagonals, so the fit must run cleanly and keep
# AGHQ-40 accuracy.
@testset "Poisson (1 + x | g), uncentred x and ρ = 0.95: stable (#834)" begin
    rng834 = StableRNG(20260629)   # version-stable: see _dgp834
    G = 100; m = 20; n = G * m; g = repeat(1:G, inner = m)
    x = 27 .+ 2 .* randn(rng834, n)
    sd0 = 0.5; sd1 = 0.02; ρ = 0.95
    Σ = [sd0^2 ρ*sd0*sd1; ρ*sd0*sd1 sd1^2]
    B = cholesky(Symmetric(Σ)).L * randn(rng834, 2, G)
    y = Float64.([rand(rng834, _Dd834.Poisson(exp(-0.5 + 0.05 * x[i] + B[1, g[i]] + B[2, g[i]] * x[i]))) for i in 1:n])
    fit = drm(bf(@formula(y ~ x + (1 + x | g))), Poisson(); data = (; y, x, g))
    θ̂ = fit.theta
    @test all(isfinite, θ̂) && isfinite(loglik(fit))
    lf = [_Dd834.logfactorial(Int(v)) for v in y]
    ll = (i, η) -> y[i] * η - exp(η) - lf[i]
    ref = _ref_loglik834(ll, g, θ̂[1] .+ θ̂[2] .* x, x, θ̂[3:5])
    @test abs(loglik(fit) - ref) < 0.05
    @test _rho834(θ̂[3:5]) > 0.5                            # strong positive correlation recovered
    # helper at the TRUE near-singular covariance (vc convention L = [e^a 0; cc e^b])
    θt = [-0.5, 0.05, log(sd0), log(sd1 * sqrt(1 - ρ^2)), ρ * sd1]
    Lt = DRModels._corr_ranef_L(θt[3], θt[4], θt[5])
    @test Lt * Lt' ≈ Σ
    Si = inv(Σ); η0 = θt[1] .+ θt[2] .* x; Zre = hcat(ones(n), x)
    for j in (1, 2, 3)
        idx = findall(==(j), g)
        hK(K) = DRModels._aghq_group_logint(ll, idx, η0, Zre, Lt, DRModels._AGHQRule(2, K), zeros(2))[1]
        h(b) = sum(ll(i, η0[i] + b[1] + b[2] * x[i]) for i in idx) - 0.5 * dot(b, Si * b) - log(2π) - 0.5 * logdet(Σ)
        r40 = _ref_group834(h; K = 40)
        @test hK(40) ≈ r40 atol = 1e-6
        @test abs(hK(DRModels._CORR_RANEF_AGHQ_K) - r40) < 1e-3
    end
end
