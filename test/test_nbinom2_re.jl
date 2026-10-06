# Random intercept on the negative-binomial (NB2) mean: an overdispersed count
# GLMM, y ~ x + (1|g) with sigma ~ 1. log μ_i = Xμ_iᵀβ + b_{g(i)}, b_g ~ N(0,σ_b²),
# dispersion θ. The group effect is integrated out per group by Gauss–Hermite
# quadrature. Recovery: β slope, dispersion θ, σ_b.
using DRModels
using Test, Random
import Distributions

@testset "NB2 random intercept (1|g) — recovery" begin
    Random.seed!(20260628)
    G = 45; m = 30; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.4, 0.5]; θ = 3.0; σb = 0.5
    bg = σb .* randn(G)
    μ = exp.(β[1] .+ β[2] .* x .+ bg[g])
    y = Float64.([rand(Distributions.NegativeBinomial(θ, θ / (θ + μ[i]))) for i in 1:n])
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.10       # log-mean slope
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ θ atol = 1.2   # dispersion θ
    @test re_sd(fit)[:g] ≈ σb atol = 0.20            # group random-intercept SD
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)
end
