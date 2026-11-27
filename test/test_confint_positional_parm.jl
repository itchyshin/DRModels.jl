# #1593: positional second argument must not surface a bare MethodError.
using DRModels
using Test
using Random
using StatsModels: @formula

@testset "#1593 confint positional block name" begin
    Random.seed!(20261127)
    n = 50
    y1, y2 = randn(n), randn(n)
    form = bf(mu1 = @formula(y1 ~ 1), mu2 = @formula(y2 ~ 1),
              sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
              rho12 = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = (; y1, y2))
    @test confint(fit; parm = :rho12) isa Vector
    err = try
        confint(fit, :rho12)
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("parm", msg)
    @test occursin("confint(fit; parm = :rho12)", msg)
end