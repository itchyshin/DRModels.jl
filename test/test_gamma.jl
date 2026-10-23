# Gamma family: strictly-positive continuous responses (durations, sizes,
# concentrations). Log link on the mean μ; the `sigma` slot carries σ = the
# coefficient of variation, mapped to the shape α = 1/σ² (var = μ²σ²). Fixed
# effects, ML. `Distributions.Gamma` is qualified — DRModels has its own family type.
using DRModels
using Test, Random
import Distributions

@testset "Gamma (positive continuous, log link) — recovery" begin
    Random.seed!(20260618)
    n = 3000
    x = randn(n)
    β = [0.5, 0.4]; α = 8.0                         # log μ = 0.5 + 0.4x; shape α
    μ = exp.(β[1] .+ β[2] .* x)
    y = Float64.([rand(Distributions.Gamma(α, μi / α)) for μi in μ])   # mean = α·scale = μ
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.06       # log-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.06       # log-mean slope
    α̂ = exp(-2 * coef(fit, :sigma)[1])               # shape α = 1/σ²
    @test α̂ ≈ α atol = 2.0                            # shape — weakly identified
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)                      # fitted means are positive
end
