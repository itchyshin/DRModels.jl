# DRModels.jl#1531: ZeroOneBeta `(1 | g)` must not abort on a bad Hessian when `se = false`.
using DRModels
using Random
using Test

@testset "ZeroOneBeta (1|g): se=false survives bad Hessian (#1531)" begin
    Random.seed!(11)
    n = 50
    g = repeat(1:5, inner = 10)
    x = randn(n)
    y = clamp.(rand(n), 0.05, 0.95)
    data = (; y, x, g)
    f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))

    fit_ctl = drm(f, ZeroOneBeta(); data = data, control = DRMControl(se = false))
    fit_kw = drm(f, ZeroOneBeta(); data = data, se = false)
    for fit in (fit_ctl, fit_kw)
        @test fit isa DrmFit
        @test isfinite(loglik(fit))
        @test all(isfinite, fit.theta)
        @test all(isnan, fit.vcov)
        @test fit.uncertainty === :skipped
    end

    fit_fe = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), ZeroOneBeta(); data = (; y, x), se = false)
    @test isfinite(loglik(fit_fe))
end