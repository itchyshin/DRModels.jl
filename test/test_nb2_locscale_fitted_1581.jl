# Regression for #1581: coupled NB2 location–scale admits default fitted/residuals.
using DRModels, Random, LinearAlgebra, Distributions, Test

Random.seed!(2026)
G = 15
m = 8
n = G * m
species = repeat(1:G, inner = m)
x = randn(n)
Λ = [0.25 0.05; 0.05 0.16]
LΛ = cholesky(Symmetric(Λ)).L
A = [LΛ * randn(2) for _ in 1:G]

function _draw(i)
    r = exp(0.3 + A[species[i]][2])
    μ = exp(0.5 + 0.4 * x[i] + A[species[i]][1])
    return Float64(rand(NegativeBinomial(r, r / (r + μ))))
end

data = (; y = [_draw(i) for i in 1:n], x, species)

@testset "#1581 NB2 coupled location–scale fitted/residuals" begin
    fit = drm(
        bf(@formula(y ~ x + (1 | p | species)), @formula(sigma ~ 1 + (1 | p | species))),
        NegBinomial2(); data = data)
    @test fit.converged
    @test haskey(fit.scales, :cond_fit_eta_re)
    μ = fitted(fit)
    @test length(μ) == nobs(fit)
    @test all(isfinite, μ)
    r = residuals(fit)
    @test length(r) == nobs(fit)
    @test r ≈ data.y[1:nobs(fit)] .- μ
    μ0 = fitted(fit; re = :zero)
    @test length(μ0) == length(μ)
    @test all(isfinite, μ0)
    @test predict(fit, data) ≈ μ0 rtol = 1e-6
end