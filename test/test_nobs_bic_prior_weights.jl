# Prior weights are not frequency weights: BIC uses nobs (row count), drmTMB #1863.
using Test
using DRModels, Random

@testset "prior weights: nobs row count and BIC vs row replication (drmTMB #1863)" begin
    Random.seed!(2)
    d = (; y = randn(30), x = randn(30))
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))
    ctrl = DRMControl(se = false)
    f_unit = drm(f, Gaussian(); data = d, control = ctrl)
    f_w = drm(f, Gaussian(); data = d, weights = fill(2.0, 30), control = ctrl)
    idx = repeat(1:30, inner = 2)
    d_rep = (; y = d.y[idx], x = d.x[idx])
    f_rep = drm(f, Gaussian(); data = d_rep, control = ctrl)

    @test nobs(f_unit) == 30
    @test nobs(f_w) == 30
    @test nobs(f_rep) == 60
    @test loglik(f_w) ≈ loglik(f_rep) atol = 1e-10
    @test aic(f_w) ≈ aic(f_rep) atol = 1e-10
    @test bic(f_w) != bic(f_rep)
    @test bic(f_w) < bic(f_rep)  # fewer rows => smaller log(n) penalty term
end