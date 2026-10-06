# Random intercept on the Binomial mean: a logistic GLMM, cbind(s, f) ~ x + (1|g).
# logit μ_i = Xμ_iᵀβ + b_{g(i)}, b_g ~ N(0,σ_b²). No closed-form marginal — the
# group effect is integrated out per group by 32-node Gauss–Hermite quadrature
# (the same machinery as the Poisson/Beta σ-RE). Recovery: β + σ_b.
using DRModels
using Test, Random
import Distributions

@testset "Binomial logistic GLMM (1|g) — σ_b recovery" begin
    Random.seed!(20260601)
    G = 60; m = 20; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.2, 0.6]; σb = 0.6
    bg = σb .* randn(G)
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x .+ bg[g])))
    ntr = fill(20, n)                                   # 20 trials per group-obs
    s = [rand(Distributions.Binomial(ntr[i], μ[i])) for i in 1:n]
    fail = ntr .- s
    data = (; s = Float64.(s), fail = Float64.(fail), x, g)

    fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.15          # logit-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08          # logit-mean slope
    @test re_sd(fit)[:g] ≈ σb atol = 0.15               # group random-intercept SD
    @test isfinite(loglik(fit))
    @test fit.converged
    @test all(0 .< fitted(fit) .< 1)
    # correlated slope (1 + x | g) is now supported (#753) — see test_binomial_slope_re.jl;
    # independent slope (0 + x | g) remains out of scope here — must be rejected
    @test_throws ErrorException drm(bf(@formula(cbind(s, fail) ~ x + (0 + x | g))), Binomial(); data = data)
end
