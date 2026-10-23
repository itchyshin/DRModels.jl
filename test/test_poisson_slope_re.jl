# Correlated random intercept + slope on a non-Gaussian mean: a Poisson GLMM with
# (1 + x | g). Per group (b0,b1) ~ N(0, Σ); log λ_i = Xμ_iᵀβ + b0_g + b1_g·x_i.
# Because groups are disjoint the per-group 2-D integral factorises, so it is done
# by a 2-D Gauss–Hermite tensor grid. Recovery: β slope and the RE covariance Σ.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "Poisson correlated random slope (1+x|g) — recovery" begin
    Random.seed!(20260701)
    G = 120; m = 25; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.3, 0.4]; sd0 = 0.5; sd1 = 0.4; ρ = 0.3
    Σ = [sd0^2 ρ*sd0*sd1; ρ*sd0*sd1 sd1^2]
    B = cholesky(Symmetric(Σ)).L * randn(2, G)          # 2×G correlated (b0,b1) per group
    b0 = B[1, :]; b1 = B[2, :]
    λ = exp.(β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x)
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 + x | g))), Poisson(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.12          # population log-rate slope
    V = vc(fit)[:g]                                       # 2×2 RE covariance
    @test sqrt(V[1, 1]) ≈ sd0 atol = 0.20                # intercept-RE SD
    @test sqrt(V[2, 2]) ≈ sd1 atol = 0.20                # slope-RE SD
    @test isfinite(loglik(fit))
end
