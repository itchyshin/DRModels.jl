# #1559 / #1560: predict and simulate must refuse zero-row or ragged `newdata`
# before design construction, not with a raw matmul DimensionMismatch.
using DRModels
using Test
using Random
using StatsModels: @formula

@testset "#1559 zero-row newdata" begin
    Random.seed!(261126)
    n = 25
    x = randn(n)
    y = randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    nd = (; x = Float64[])
    @test_throws ArgumentError predict(fit, nd)
    err = try
        predict(fit, nd)
    catch e
        e
    end
    @test occursin("no rows", sprint(showerror, err))
    @test_throws ArgumentError simulate(fit; newdata = nd, nsim = 1)
end

@testset "#1560 ragged vector columns in newdata" begin
    Random.seed!(3126)
    n = 25
    x = randn(n)
    y = randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    # Scalar metadata is ignored (#1548); mismatch must be between vector columns.
    nd_ok = (; x = [0.5, 1.0], n = 1)
    @test length(predict(fit, nd_ok)) == 2
    nd_bad = (; x = [0.5, 1.0], id = [1.0])
    err = try
        predict(fit, nd_bad)
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("2 row", msg)
    @test occursin("`id` has 1", msg)
    @test_throws ArgumentError simulate(fit; newdata = nd_bad, nsim = 1)
end