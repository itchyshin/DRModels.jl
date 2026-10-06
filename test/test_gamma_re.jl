# Random intercept on the Gamma mean: a positive-continuous GLMM, y ~ x + (1|g).
# log μ_i = Xμ_iᵀβ + b_{g(i)}, b_g ~ N(0,σ_b²), shape α = 1/σ². The group effect
# is integrated out per group by Gauss–Hermite quadrature. Recovery: β, α, σ_b.
using DRModels
using Test, Random
import Distributions

@testset "Gamma random intercept (1|g) — recovery" begin
    Random.seed!(20260630)
    G = 50; m = 30; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.5, 0.4]; α = 8.0; σb = 0.4
    bg = σb .* randn(G)
    μ = exp.(β[1] .+ β[2] .* x .+ bg[g])
    y = Float64.([rand(Distributions.Gamma(α, μ[i] / α)) for i in 1:n])
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gamma(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.10       # log-mean slope
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ α atol = 3.0   # shape α
    @test re_sd(fit)[:g] ≈ σb atol = 0.15            # group random-intercept SD
    @test isfinite(loglik(fit))
end
