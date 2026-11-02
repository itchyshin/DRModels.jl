# #974: dense known sampling covariance `V = V` is bivariate-only. On the
# univariate Gaussian route it must be a named ArgumentError that points to
# `meta_vcov_bivariate` and `meta_V(vi)`, not a bare MethodError.
using DRModels
using Test
using Random
using LinearAlgebra

@testset "#974: dense V on the univariate Gaussian route is refused" begin
    rng = MersenneTwister(974)
    n = 10
    x = randn(rng, n)
    vi = 0.05 .+ 0.05 .* rand(rng, n)
    y = 0.3 .+ 0.5 .* x .+ sqrt.(vi .+ 0.04) .* randn(rng, n)
    V = Matrix(Diagonal(vi))
    V[1:2, 1:2] .= [vi[1] 0.01; 0.01 vi[2]]          # block-diagonal, not diagonal
    data = (; y, x, vi)

    f = bf(@formula(y ~ x), @formula(sigma ~ 1))
    err = try drm(f, Gaussian(); data = data, V = V); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("meta_vcov_bivariate", sprint(showerror, err))
    @test occursin("meta_V", sprint(showerror, err))
    @test_throws ArgumentError drm(f, Gaussian(); data = data, V = V)

    # The supported diagonal route is unchanged.
    fd = drm(bf(@formula(y ~ x + meta_V(vi)), @formula(sigma ~ 1)), Gaussian(); data = data)
    @test isfinite(loglik(fd))
end
