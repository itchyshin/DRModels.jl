# #1592: zero-row `newdata` must be refused before design construction (#1559 / #1560).
using DRModels
using Test
using Random
using StatsModels: @formula

@testset "#1592 predict on empty newdata" begin
    Random.seed!(271127)
    n = 20
    x, y = randn(n), randn(n)
    fit = drm(bf(@formula(y ~ x)), Gaussian(); data = (; y, x))
    nd = (; x = Float64[])
    @test_throws ArgumentError predict(fit, nd)
    err = try
        predict(fit, nd)
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("no rows", msg)
    @test !occursin("DivideError", msg)
end