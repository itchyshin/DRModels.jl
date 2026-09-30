# Correlated random intercept + slope on the Binomial mean (#753): a logistic
# GLMM with (1 + x | g). Per group (b0,b1) ~ N(0, Σ); logit μ_i =
# Xμ_iᵀβ + b0_g + b1_g·x_i. Because groups are disjoint the per-group 2-D
# integral factorises; it is done by per-group ADAPTIVE Gauss–Hermite
# quadrature (`_aghq_marginal_loglik`, #834), the same scheme as
# BetaBinomial's `_fit_betabinomial_corr_ranef` minus the precision φ — this
# replaces the original #753 fixed 12×12 prior-scale grid, which was ~15 nat
# off on this file's own DGP (see #834). Recovery: population slope β, both
# RE SDs. Also pins the refusals that remain: `(0 + x | g)`, `marginal = :VA`,
# and a slope predictor constant within every group (unidentified slope SD /
# correlation).
using DRModels
using Test, Random, LinearAlgebra, ForwardDiff
import Distributions

@testset "Binomial correlated random slope (1+x|g) — recovery" begin
    Random.seed!(20260627)
    G = 150; m = 20; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.2, 0.5]; sd0 = 0.5; sd1 = 0.4; ρ = 0.3
    Σ = [sd0^2 ρ*sd0*sd1; ρ*sd0*sd1 sd1^2]
    B = cholesky(Symmetric(Σ)).L * randn(2, G)           # 2×G correlated (b0,b1) per group
    b0 = B[1, :]; b1 = B[2, :]
    ntr = fill(20, n)                                    # 20 trials each
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x)))
    s = [rand(Distributions.Binomial(ntr[i], μ[i])) for i in 1:n]
    fail = ntr .- s
    data = (; s = Float64.(s), fail = Float64.(fail), x, g)

    fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 + x | g))), Binomial(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15           # population logit-mean slope
    V = vc(fit)[:g]                                       # 2×2 RE covariance
    @test sqrt(V[1, 1]) ≈ sd0 atol = 0.20                # intercept-RE SD
    @test sqrt(V[2, 2]) ≈ sd1 atol = 0.20                # slope-RE SD
    @test isfinite(loglik(fit))
    @test fit.converged
    @test all(0 .< fitted(fit) .< 1)                     # fitted mean success probabilities

    # independent random slope (0 + x | g) is still out of scope
    @test_throws ErrorException drm(bf(@formula(cbind(s, fail) ~ x + (0 + x | g))), Binomial(); data = data)

    # marginal = :VA has no public route for the correlated slope
    @test_throws ArgumentError drm(bf(@formula(cbind(s, fail) ~ x + (1 + x | g))), Binomial();
                                    data = data, marginal = :VA)

    # slope predictor constant within every group → slope SD / correlation unidentified
    xg = Float64.(g)                                      # constant within each level of g
    data_const = (; s = Float64.(s), fail = Float64.(fail), x = xg, g)
    @test_throws ErrorException drm(bf(@formula(cbind(s, fail) ~ x + (1 + x | g))), Binomial();
                                     data = data_const)

    # --- AGHQ-40 cross-check (#834 follow-up): the DEFAULT nq (_CORR_RANEF_AGHQ_K)
    # logLik must agree with an INDEPENDENT AGHQ-40 implementation (own Newton mode
    # finder, own 40×40 tensor grid — not DRModels' `_aghq_group_logint`) evaluated
    # at the fit's own θ̂. Mirrors test/test_adaptive_ghq.jl's Poisson/Beta-binomial
    # checks (#839); duplicated here (not `include`d) because runtests.jl shards
    # each test file into its own process.
    function _gh_bslope(K)
        βn = [sqrt(k / 2) for k in 1:(K-1)]
        E = eigen(SymTridiagonal(zeros(K), βn))
        return E.values, sqrt(π) .* E.vectors[1, :] .^ 2
    end
    function _ref_group_bslope(h; K = 40)
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
        K == 1 && return h(b) + log(2π) - 0.5 * logdet(Hm)     # exact Laplace formula
        C = cholesky(Symmetric(inv(Hm))).L
        z, w = _gh_bslope(K); lw = log.(w)
        terms = [lw[j] + lw[k] + h(b + sqrt(2) * C * [z[j], z[k]]) + z[j]^2 + z[k]^2 for j in 1:K for k in 1:K]
        mx = maximum(terms)
        return log(2) + logdet(C) + mx + log(sum(exp.(terms .- mx)))
    end
    function _ref_loglik_bslope(llf, gg, η0, xx, θre; K = 40)
        L = [exp(θre[1]) 0; θre[3] exp(θre[2])]; S = L * L'; Si = inv(S); ldS = logdet(S)
        tot = 0.0
        for j in 1:maximum(gg)
            idx = findall(==(j), gg)
            h(b) = sum(llf(i, η0[i] + b[1] + b[2] * xx[i]) for i in idx) - 0.5 * dot(b, Si * b) - log(2π) - 0.5 * ldS
            tot += _ref_group_bslope(h; K)
        end
        return tot
    end

    θ̂ = fit.theta
    llf = (i, η) -> (μi = 1 / (1 + exp(-η)); Distributions.logpdf(Distributions.Binomial(Int(ntr[i]), μi), Int(s[i])))
    ref40 = _ref_loglik_bslope(llf, g, θ̂[1] .+ θ̂[2] .* x, x, θ̂[3:5])
    @test abs(loglik(fit) - ref40) < 0.05                # default nq (=5): measured error ≈ 5.5e-4 nat

    # --- K = 1 is exactly Laplace (drmTMB's own integrator): fit at nq = 1 and
    # cross-check against the independent Laplace formula (K = 1 branch above) at
    # the SAME θ̂ — this is the internal analogue of "Laplace ↔ Laplace" parity
    # with drmTMB 0.7.1 (measured separately on the same simulated data: logLik
    # agrees to 7e-10, β to 2e-9 nat/logit-units; see the after-task report).
    fit1 = DRModels._fit_binomial_corr_ranef(DRModels.Binomial(), s, ntr, hcat(ones(n), x), x,
                                              g, G, ["(Intercept)", "x"], :g, 1e-8; nq = 1)
    θ̂1 = fit1.theta
    ref_laplace = _ref_loglik_bslope(llf, g, θ̂1[1] .+ θ̂1[2] .* x, x, θ̂1[3:5]; K = 1)
    @test loglik(fit1) ≈ ref_laplace atol = 1e-6
end
