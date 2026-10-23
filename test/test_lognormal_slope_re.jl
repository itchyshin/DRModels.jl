# Correlated random intercept + slope on the LogNormal log-mean: a positive-
# continuous GLMM, y ~ x + (1 + x | g) with log y Gaussian. μ_i = Xμ_iᵀβ +
# b0_{g(i)} + b1_{g(i)}·x_i is the mean of log y, with (b0,b1) ~ N(0, Σ) per group,
# σ = SD of log y. The 2-D group effect is integrated out by K×K Gauss–Hermite
# quadrature. Recovery: log-mean slope and the RE covariance Σ.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "LogNormal correlated random slope (1+x|g) — recovery" begin
    Random.seed!(20260802)
    G = 120; m = 25; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.5, 0.4]; σlog = 0.3
    sd0 = 0.5; sd1 = 0.4; ρ = 0.3
    Σ = [sd0^2 ρ*sd0*sd1; ρ*sd0*sd1 sd1^2]
    B = cholesky(Symmetric(Σ)).L * randn(2, G)       # 2×G correlated (b0,b1) per group
    b0 = B[1, :]; b1 = B[2, :]
    μ = β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x     # mean of log y
    y = exp.(μ .+ σlog .* randn(n))                  # log y ~ N(μ, σlog)

    fit = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), LogNormal(); data = (; y, x, g))

    @test coef(fit, :mu)[2] ≈ 0.4 atol = 0.15        # log-mean slope
    V = vc(fit)[:g]                                   # 2×2 RE covariance
    @test sqrt(V[1, 1]) ≈ sd0 atol = 0.2             # intercept-RE SD
    @test sqrt(V[2, 2]) ≈ sd1 atol = 0.2             # slope-RE SD
    @test isfinite(loglik(fit))
end
