# Random intercept on the Beta-binomial mean: an overdispersed-binomial GLMM,
# cbind(s, f) ~ x + (1|g). logit μ_i = Xμ_iᵀβ + b_{g(i)}, b_g ~ N(0,σ_b²), with
# precision φ = 1/σ². The group effect is integrated out per group by 32-node
# Gauss–Hermite quadrature. Recovery: β slope, φ, and σ_b.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "Beta-binomial random intercept (1|g) — recovery" begin
    Random.seed!(20260630)
    G = 80; m = 30; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.3, 0.6]; φ = 12.0; σb = 0.5                   # logit μ = 0.3 + 0.6x + b_g; precision φ
    bg = σb .* randn(G)
    ntr = fill(20, n)                                    # 20 trials each
    μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x .+ bg[g])))
    s = [rand(Distributions.BetaBinomial(ntr[i], μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n]
    fail = ntr .- s
    data = (; s = Float64.(s), fail = Float64.(fail), x, g)

    fit = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sigma ~ 1)), BetaBinomial(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.12           # logit-mean slope
    @test exp(-2 * coef(fit, :sigma)[1]) ≈ φ atol = 5.0   # precision φ — weakly identified
    @test re_sd(fit)[:g] ≈ σb atol = 0.15                # group random-intercept SD
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)                     # fitted mean success probabilities
end
