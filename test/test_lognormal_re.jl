# Random intercept on the LogNormal log-mean: a positive-continuous GLMM,
# y ~ x + (1|g) with log y Gaussian. μ_i = Xμ_iᵀβ + b_{g(i)} is the mean of log y,
# b_g ~ N(0,σ_b²), σ = SD of log y. The group effect is integrated out by
# Gauss–Hermite quadrature. Recovery: β, σ, σ_b.
using DRModels
using Test, Random
import Distributions

@testset "LogNormal random intercept (1|g) — recovery" begin
    Random.seed!(20260801)
    G = 40; m = 30; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.5, 0.4]; σlog = 0.3; σb = 0.5
    bg = σb .* randn(G)
    μ = β[1] .+ β[2] .* x .+ bg[g]                   # mean of log y, with group intercept
    y = exp.(μ .+ σlog .* randn(n))                  # log y ~ N(μ, σlog)
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), LogNormal(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.10       # log-mean slope
    @test exp(coef(fit, :sigma)[1]) ≈ σlog atol = 0.10   # SD of log y
    @test re_sd(fit)[:g] ≈ σb atol = 0.15            # group random-intercept SD
    @test isfinite(loglik(fit))
end
