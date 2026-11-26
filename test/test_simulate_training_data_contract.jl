# #1549: doc/API contract — marginal training-row simulation on `(1 | g)` fits
# needs `simulate(fit; data = <fitting table>)`; bare `simulate(fit)` refuses.
using DRModels
using Test
using Random
import Distributions
using StatsModels: @formula

@testset "#1549 simulate on training rows with random intercepts" begin
    Random.seed!(112608)
    G, m = 8, 20
    g = repeat(1:G, inner = m)
    n = G * m
    x = randn(n)
    y = Float64[rand(Distributions.Poisson(exp(0.2 + 0.5 * x[i]))) for i in 1:n]
    data = (; y, x, g)
    fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)

    @test_throws ArgumentError simulate(fit)
    ysim = simulate(fit; data = data, re = :marginal)
    @test length(ysim) == fit.nobs
    @test all(ysim .>= 0)

    fit0 = drm(bf(@formula(y ~ x)), Poisson(); data = data)
    @test length(simulate(fit0)) == fit0.nobs
end