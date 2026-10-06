# Correlated random intercept+slope on the negative-binomial (NB2) mean: an
# overdispersed count GLMM, y ~ x + (1 + x | g) with sigma ~ 1. log μ_i =
# Xμ_iᵀβ + b0_{g(i)} + b1_{g(i)}·x_i, (b0,b1) ~ N(0, Σ), dispersion θ. The 2-D
# group effect is integrated out per group by tensor-product Gauss–Hermite
# quadrature. Recovery: β slope and the two RE SDs.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "NB2 correlated random slope (1+x|g) — recovery" begin
    Random.seed!(20260702)
    G = 120; m = 25; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.3, 0.4]; θ = 3.0
    sd0 = 0.5; sd1 = 0.4; ρ = 0.3
    Σ = [sd0^2 ρ*sd0*sd1; ρ*sd0*sd1 sd1^2]
    B = cholesky(Symmetric(Σ)).L * randn(2, G)       # 2×G correlated (b0,b1) per group
    b0 = B[1, :]; b1 = B[2, :]
    μ = exp.(β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x)
    y = Float64.([rand(Distributions.NegativeBinomial(θ, θ / (θ + μ[i]))) for i in 1:n])

    fit = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), NegBinomial2(); data = (; y, x, g))

    @test coef(fit, :mu)[2] ≈ 0.4 atol = 0.15        # log-mean slope
    V = vc(fit)[:g]                                   # 2×2 RE covariance
    @test sqrt(V[1, 1]) ≈ sd0 atol = 0.22             # intercept-RE SD
    @test sqrt(V[2, 2]) ≈ sd1 atol = 0.22             # slope-RE SD
    @test isfinite(loglik(fit))
end
