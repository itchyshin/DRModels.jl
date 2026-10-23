# Prediction on new data: population-level Xβ̂ (random/structured effects
# integrated out). Deterministic checks.
using DRModels
using Test, Random

@testset "predict on new data" begin
    Random.seed!(2)
    n = 400
    x = randn(n)
    y = 1.0 .+ 0.5 .* x .+ 0.4 .* randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))

    β = coef(fit, :mu)
    ŷ = predict(fit, (; x = [0.0, 1.0, 2.0]))
    @test ŷ ≈ β[1] .+ β[2] .* [0.0, 1.0, 2.0]

    # in-sample prediction matches fitted
    @test predict(fit, (; x = x)) ≈ fitted(fit)

    # bivariate: per-response predictions
    y2 = -0.5 .+ 0.3 .* x .+ 0.4 .* randn(n)
    bv = drm(bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
               rho12 = @formula(rho12 ~ 1)), Gaussian(); data = (; y, y2, x))
    pm = predict(bv, (; x = [0.0, 1.0]))
    @test haskey(pm, :mu1) && haskey(pm, :mu2)
    @test length(pm[:mu1]) == 2
end
