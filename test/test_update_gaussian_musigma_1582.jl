# Regression for #1582: update() on Gaussian mean + sigma RE must not force σ-only :Laplace.
using DRModels, Random, Test

@testset "#1582 update Gaussian location–scale with RE on mu" begin
    Random.seed!(19)
    G = 8
    m = 4
    n = G * m
    data = (; y = randn(n) .+ randn(n), x = randn(n), g = repeat(1:G, inner = m))
    form = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g)))
    fit = drm(form, Gaussian(); data = data)
    @test fit.marginal === :Laplace
    up = update(fit, form; data = data)
    @test up isa DrmFit
    @test up.marginal === :Laplace
    @test length(coef(up)) == length(coef(fit))
    @test coef(up) ≈ coef(fit) rtol = 1e-5
    @test loglik(up) ≈ loglik(fit) rtol = 1e-8
end