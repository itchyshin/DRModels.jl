# cov lane 1125n6 — aghq_1d.jl contract, guards, and Laplace identity.
using DRModels
using Test

@testset "cov 1125n6: _aghq_require_1d" begin
    @test DRModels._aghq_require_1d(1) === nothing
    @test_throws ArgumentError DRModels._aghq_require_1d(2)
    @test_throws ArgumentError DRModels._aghq_require_1d(0)
end

@testset "cov 1125n6: _aghq_1d_logint quadratic" begin
    mode = 1.25
    hess = -2.0
    logf(x) = 0.3 - (x - mode)^2
    lap = logf(mode) + 0.5 * log(2π) - 0.5 * log(-hess)
    int5 = DRModels._aghq_1d_logint(logf, mode, hess; k = 5)
    int1 = DRModels._aghq_1d_logint(logf, mode, hess; k = 1)
    @test int5 ≈ lap rtol = 1e-3
    @test int1 ≈ lap rtol = 1e-10
end

@testset "cov 1125n6: _aghq_1d_logint refinements" begin
    logf(x) = -(x - 0.5)^4
    mode = 0.5
    hess = -12 * 0.5^2
    a = DRModels._aghq_1d_logint(logf, mode, hess; k = 3)
    b = DRModels._aghq_1d_logint(logf, mode, hess; k = 7)
    @test isfinite(a) && isfinite(b)
    @test b >= a - 1e-6
end

@testset "cov 1125n6: _aghq_1d argument errors" begin
    @test_throws ArgumentError DRModels._aghq_1d_logint(x -> 0.0, 0.0, 1.0; k = 0)
    @test_throws ArgumentError DRModels._aghq_1d_logint(x -> 0.0, 0.0, 2.0)
    @test isfinite(DRModels._aghq_1d_logint(x -> -x^2, 0.0, -1.0; k = 3))
    z, w = DRModels._gauss_hermite(3)
    @test_throws ArgumentError DRModels._aghq_1d_logint(x -> 0.0, 0.0, -2.0, z[1:2], w)
    @test_throws ArgumentError DRModels._aghq_1d_logint(x -> 0.0, [0.0, 1.0], -1.0)
end

@testset "cov 1125n6: _aghq_1d vector mode dispatches to scalar" begin
    logf(x) = -x^2 / 2
    v = DRModels._aghq_1d_logint(logf, [0.0], -1.0; k = 5)
    s = DRModels._aghq_1d_logint(logf, 0.0, -1.0; k = 5)
    @test v ≈ s rtol = 1e-12
end