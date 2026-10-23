# Poisson family: count responses with a log link on the mean (λ = exp(Xμβ)).
# No dispersion parameter — for that, see the negative-binomial family. Fixed
# effects, maximum likelihood. Mirrors drmTMB's `poisson`.
using DRModels
using Test, Random
import Distributions          # qualified — DRModels exports its own `Poisson` family

@testset "Poisson (counts, log link) — recovery" begin
    Random.seed!(20260615)
    n = 3000
    x = randn(n)
    β = [0.3, 0.5]                       # log λ = 0.3 + 0.5 x
    λ = exp.(β[1] .+ β[2] .* x)
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
    data = (; y, x)

    fit = drm(bf(@formula(y ~ x)), Poisson(); data = data)

    @test coef(fit, :mu)[1] ≈ β[1] atol = 0.06     # log-mean intercept
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.06     # log-mean slope
    @test isfinite(loglik(fit))
    @test all(fitted(fit) .> 0)                    # fitted means are on the response (count) scale
end
