# Correlated random intercept + slope on the Student-t mean: a robust GLMM with
# (1 + x | g). Per group (b0,b1) ~ N(0, Σ); μ_i = Xμ_iᵀβ + b0_g + b1_g·x_i
# (identity link), with the scale σ and df ν as fixed effects. Because groups are
# disjoint the per-group 2-D integral factorises, so it is done by a 2-D
# Gauss–Hermite tensor grid. Recovery: β slope, σ, ν, and the RE covariance Σ.
using DRModels
using Test, Random, LinearAlgebra
import Distributions
using Distributions: TDist

@testset "Student-t correlated random slope (1+x|g) — recovery" begin
    Random.seed!(20260632)
    G = 120; m = 25; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    β = [0.4, 0.6]; σ = 0.6; ν = 6.0
    sd0 = 0.6; sd1 = 0.45; ρ = 0.3
    Σ = [sd0^2 ρ*sd0*sd1; ρ*sd0*sd1 sd1^2]
    B = cholesky(Symmetric(Σ)).L * randn(2, G)        # 2×G correlated (b0,b1) per group
    b0 = B[1, :]; b1 = B[2, :]
    μ = β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x
    y = μ .+ σ .* rand(TDist(ν), n)                   # location-scale t with a group intercept+slope
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = data)

    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.12        # population mean slope
    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.10   # scale σ
    @test 2 + exp(coef(fit, :nu)[1]) ≈ ν atol = 3.0   # ν = 2 + exp(η); df weakly identified — loose
    V = vc(fit)[:g]                                    # 2×2 RE covariance
    @test sqrt(V[1, 1]) ≈ sd0 atol = 0.20             # intercept-RE SD
    @test sqrt(V[2, 2]) ≈ sd1 atol = 0.20             # slope-RE SD
    @test isfinite(loglik(fit))
end
