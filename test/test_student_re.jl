# Random intercept on the Student-t mean: a robust location–scale GLMM,
# y ~ x + (1|g). μ_i = Xμ_iᵀβ + b_{g(i)} (identity link), b_g ~ N(0,σ_b²); the
# scale σ and degrees of freedom ν are fixed effects. The group effect is
# integrated out per group by Gauss–Hermite quadrature. Recovery: β, σ, ν, σ_b.
using DRModels
using Test, Random, LinearAlgebra
import Distributions
using Distributions: TDist

@testset "Student-t random intercept (1|g) — recovery" begin
    Random.seed!(20260631)
    G = 40; m = 30; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.5, -0.4]; σ = 0.8; ν = 5.0; σb = 0.5
    bg = σb .* randn(G)
    μ = β[1] .+ β[2] .* x .+ bg[g]
    y = μ .+ σ .* rand(TDist(ν), n)                  # location-scale t with a group intercept
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.10        # population mean slope
    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.12   # scale σ
    @test 2 + exp(coef(fit, :nu)[1]) ≈ ν atol = 2.5   # ν = 2 + exp(η); df weakly identified — loose
    @test re_sd(fit)[:g] ≈ σb atol = 0.15             # group random-intercept SD
    @test isfinite(loglik(fit))
end
