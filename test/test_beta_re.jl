# Random intercept on the beta mean: a proportion GLMM, y ~ x + (1|g). logit μ_i
# = Xμ_iᵀβ + b_{g(i)}, b_g ~ N(0,σ_b²), precision φ = 1/σ². The group effect is
# integrated out per group by Gauss–Hermite quadrature. Recovery: β slope, φ, σ_b.
using DRModels
using Test, Random
import Distributions

@testset "Beta random intercept (1|g) — recovery" begin
    Random.seed!(20260629)
    G = 50; m = 30; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.2, 0.6]; φ = 15.0; σb = 0.5
    bg = σb .* randn(G)
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x .+ bg[g])))
    y = Float64.([rand(Distributions.Beta(μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n])
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Beta(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.12       # logit-mean slope
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ φ atol = 6.0   # precision φ
    @test re_sd(fit)[:g] ≈ σb atol = 0.20            # group random-intercept SD
    @test isfinite(loglik(fit))
end
