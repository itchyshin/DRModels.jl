# refs #1616 — pin today's definition: nobs counts fitted rows, not sum(prior weights).
using Random, LinearAlgebra, StatsModels, StatsAPI, Test
using DRModels

@testset "nobs with prior weights > 1 (refs #1616)" begin
    Random.seed!(2711)
    BLAS.set_num_threads(1)
    n = 24
    g = repeat(1:4, 6)
    x = randn(n)
    y = 0.4 .* x .+ randn(n)
    w = fill(1.0, n)
    w[1] = 3.0

    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
              data = (; y, x, g), weights = w)

    @test nobs(fit) == n
    @test sum(weights(fit)) == sum(w) ≈ 26.0
    k = length(fit.theta)
    @test bic(fit) ≈ -2 * loglik(fit) + k * log(nobs(fit)) atol = 1e-12
    @test aicc(fit) ≈ aic(fit) + 2 * k * (k + 1) / (nobs(fit) - k - 1) atol = 1e-12
end