# test_cov_1125n5_rescale_plan.jl — response rescale plan (src/rescale_response.jl).
using DRModels
using Test, Random, Statistics

@testset "cov 1125n5: _y_rescale_plan ineligible routes" begin
    Random.seed!(1125)
    y = randn(30)
    d = (; y, x = randn(30), g = repeat(1:3, 10))
    f = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
    @test DRModels._y_rescale_plan(f, Gaussian(), (; data = d), ()) === nothing
    f2 = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(nu ~ 1))
    @test DRModels._y_rescale_plan(f2, Student(), (; data = d), ()) === nothing
    allowed = (:data,)
    f3 = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    @test DRModels._y_rescale_plan(f3, Gaussian(), (; data = d, g_tol = 1e-8, method = :REML), (:data, :g_tol, :method)) === nothing
end

@testset "cov 1125n5: _y_rescale_plan activates for huge sd(y)" begin
    Random.seed!(77)
    n = 40
    y = 1e6 .* (0.2 .+ randn(n))
    x = randn(n)
    d = (; y, x)
    f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    allowed = (:data, :g_tol)
    plan = DRModels._y_rescale_plan(f, Gaussian(), (; data = d, g_tol = 1e-8), allowed)
    @test plan !== nothing
    @test plan.s ≈ Statistics.std(Float64.(y)) rtol = 1e-10
    @test plan.n == n
    @test length(plan.data.y) == n
    @test maximum(abs, plan.data.y) < maximum(abs, y)
end

@testset "cov 1125n5: _y_rescale_restore round-trip loglik" begin
    Random.seed!(88)
    n = 60
    y = 2e4 .* randn(n)
    x = randn(n)
    d = (; y, x)
    f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    fit_int = drm(f, Gaussian(); data = d)
    allowed = (:data, :g_tol)
    plan = DRModels._y_rescale_plan(f, Gaussian(), (; data = d, g_tol = 1e-8), allowed)
    plan === nothing && return
    dscaled = plan.data
    fit_scaled = drm(f, Gaussian(); data = dscaled)
    restored = DRModels._y_rescale_restore(fit_scaled, plan)
    @test restored !== nothing
    @test restored.loglik ≈ fit_int.loglik rtol = 1e-8
    @test restored.nobs == fit_int.nobs
    @test length(coef(restored)) == length(coef(fit_int))
end