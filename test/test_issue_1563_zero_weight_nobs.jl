# #1563: zero prior weight rows are excluded from nobs / dof_residual (drmTMB #1863 parity).
using DRModels, StatsAPI, Random, Test

@testset "#1563 zero prior weight nobs" begin
    Random.seed!(26026)
    x = randn(20)
    y = ones(Int, 20)
    w = ones(20)
    w[5] = 0.0
    idx = setdiff(1:20, 5)

    fit_w = drm(bf(@formula(y ~ x)), Poisson(); data = (; y, x), weights = w)
    fit_d = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = y[idx], x = x[idx]))

    @test loglikelihood(fit_w) ≈ loglikelihood(fit_d)
    @test coef(fit_w) ≈ coef(fit_d)
    @test nobs(fit_w) == nobs(fit_d) == 19
    @test dof_residual(fit_w) == dof_residual(fit_d)
    @test sum(weights(fit_w)) == 19.0
    @test bic(fit_w) ≈ bic(fit_d) rtol = 1e-10
end
