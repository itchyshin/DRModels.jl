# Correlated random intercept + slope on the Binomial mean (#753): a logistic
# GLMM with (1 + x | g). Per group (b0,b1) ~ N(0, Σ); logit μ_i =
# Xμ_iᵀβ + b0_g + b1_g·x_i. Because groups are disjoint the per-group 2-D
# integral factorises, so it is done by a 2-D Gauss–Hermite tensor grid — the
# same scheme as BetaBinomial's `_fit_betabinomial_corr_ranef` minus the
# precision φ. Recovery: population slope β, both RE SDs. Also pins the
# refusals that remain: `(0 + x | g)`, `marginal = :VA`, and a slope predictor
# constant within every group (unidentified slope SD / correlation).
using DRModels
using Test, Random, LinearAlgebra
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
end
