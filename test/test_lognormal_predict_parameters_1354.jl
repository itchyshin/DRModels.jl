# Refs #1354: LogNormal `predict_parameters(type = :response)[:mu]` is log-scale μ.
using DRModels
using Test

@testset "LogNormal predict_parameters response mu (#1354)" begin
    x = [sin(i) for i in 1:80]
    y = [exp(0.3 + 0.5 * x[i] + 0.4 * cos(7 * i)) for i in 1:80]
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = (; y, x))
    nd = (; x = [0.0, 1.0])
    pred = predict(fit, nd; type = :response)
    mu_pp = predict_parameters(fit, nd; type = :response)[:mu]
    @test mu_pp ≈ pred atol = 1e-10
    @test mu_pp ≈ predict(fit, nd; type = :link) atol = 1e-10
    @test all(mu_pp .< 2.0)  # not exp(mu)
end