# DRModels.jl#1577 — scalar NamedTuple predictors yield one-row predictions
using DRModels, Random, Test

@testset "issue 1577: predict scalar covariate columns" begin
    rng = MersenneTwister(1)
    n = 50
    x = randn(rng, n)
    y = randn(rng, n)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
  pred_vec = predict(fit, (; x = [0.0]))
    pred_scl = predict(fit, (; x = 0.0))
    @test length(pred_scl) == 1
    @test isapprox(pred_scl[1], pred_vec[1]; atol = 1e-10)
    pp_vec = predict_parameters(fit, (; x = [0.0]))
    pp_scl = predict_parameters(fit, (; x = 0.0))
    @test length(pp_scl[:mu]) == 1
    @test isapprox(pp_scl[:mu][1], pp_vec[:mu][1]; atol = 1e-10)
end