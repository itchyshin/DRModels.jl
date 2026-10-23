# LogNormal family: strictly-positive responses whose log is Gaussian. The mean
# formula μ is the mean of log y (identity link on the log scale); σ is the SD of
# log y (log link). logpdf = Normal(log y; μ, σ) − log y. Fixed effects, ML.
# Mirrors drmTMB's `lognormal`.
using DRModels
using Test, Random

@testset "LogNormal (positive continuous) — recovery" begin
    Random.seed!(20260621)
    n = 2000
    x = randn(n)
    β = [0.5, 0.3]; σ = 0.4                  # log y ~ N(0.5 + 0.3x, σ²)
    y = exp.(β[1] .+ β[2] .* x .+ σ .* randn(n))
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.05      # log-scale mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.05      # log-scale mean slope
    @test exp(coef(fit, :sigma)[1]) ≈ σ atol = 0.05  # SD of log y
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)                     # fitted medians exp(μ̂) > 0
end
