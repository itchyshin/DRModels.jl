# Random intercept on a non-Gaussian family's mean: a Poisson count GLMM,
# y ~ x + (1|g). log λ_i = Xμ_iᵀβ + b_{g(i)}, b_g ~ N(0,σ_b²). No closed-form
# marginal — the group effect is integrated out per group by Gauss–Hermite
# quadrature (the same machinery as the Gaussian σ-RE). Recovery: β + σ_b.
using DRModels
using Test, Random, Statistics
import Distributions

@testset "Poisson random intercept (1|g) — recovery" begin
    rng = MersenneTwister(20260627)
    G = 40; m = 30; n = G * m
    g = repeat(1:G, inner = m); x = randn(rng, n)
    β = [0.3, 0.5]; σb = 0.6
    bg = σb .* randn(rng, G)
    bg .-= mean(bg)
    λ = exp.(β[1] .+ β[2] .* x .+ bg[g])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.15      # population log-rate intercept (centered RE draw)
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08      # log-mean slope
    @test re_sd(fit)[:g] ≈ σb atol = 0.15           # group random-intercept SD
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end
