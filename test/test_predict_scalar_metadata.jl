# #1548: `predict(fit, newdata)` must tolerate scalar metadata fields in a
# column table, matching `drm(...; data)` where only vector columns matter.
using DRModels
using Test
using Random
import Distributions
using StatsModels: @formula

@testset "#1548 predict ignores scalar metadata columns" begin
    Random.seed!(112607)
    G, m = 8, 20
    g = repeat(1:G, inner = m)
    n = G * m
    x = randn(n)
    y = Float64[rand(Distributions.Poisson(exp(0.2 + 0.5 * x[i]))) for i in 1:n]
    data = (; y, x, g)
    fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)
    p0 = predict(fit, data)
    p1 = predict(fit, merge(data, (n = n,)))
    @test length(p1) == length(p0) == fit.nobs
    @test p1 ≈ p0

    μ = zeros(n)
    @test predict(fit, merge(data, (μ = μ, nobs_meta = n))) ≈ p0
end