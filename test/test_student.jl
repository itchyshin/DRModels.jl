# Student-t family: robust location–scale–shape regression. A formula per
# parameter — μ (location), σ (scale, log link), ν (degrees of freedom, log
# link). Heavy tails downweight outliers; as ν → ∞ it tends to Gaussian. Fixed
# effects, maximum likelihood. Mirrors drmTMB's `student` family.
using DRModels
using Test, Random
using Distributions: TDist

@testset "Student-t location–scale — recovery" begin
    Random.seed!(20260614)
    n = 3000
    x = randn(n)
    β = [0.5, -0.4]; σ = 0.8; ν = 5.0
    y = β[1] .+ β[2] .* x .+ σ .* rand(TDist(ν), n)   # location-scale t
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.08
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08
    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.10     # scale
    @test 2 + exp(coef(fit, :nu)[1]) ≈ ν atol = 2.5      # ν = 2 + exp(η); df weakly identified — loose
    @test isfinite(loglik(fit))
end
