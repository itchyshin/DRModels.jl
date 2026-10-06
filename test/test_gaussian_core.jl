# Univariate Gaussian location–scale, fixed effects — the drmTMB homepage model.
# Recovery test (Curie): simulate with known coefficients, fit via the public
# API, assert recovery. This is the RED test for the `drm` / `bf` / `Gaussian`
# front end.
using DRModels
using Test, Random

@testset "Gaussian location-scale (univariate, fixed effects) — recovery" begin
    Random.seed!(20260530)
    n = 4000
    x = randn(n)
    βμ = [0.5, -0.8]      # (Intercept), x   on μ
    βσ = [-0.3, 0.4]      # (Intercept), x   on log σ
    μ = βμ[1] .+ βμ[2] .* x
    logσ = βσ[1] .+ βσ[2] .* x
    y = μ .+ exp.(logσ) .* randn(n)
    data = (; y, x)

    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

    @test coef(fit, :mu) ≈ βμ atol = 0.08
    @test coef(fit, :sigma) ≈ βσ atol = 0.08
    @test isfinite(loglik(fit))
    @test size(vcov(fit), 1) == 4          # 2 μ + 2 σ coefficients
end
