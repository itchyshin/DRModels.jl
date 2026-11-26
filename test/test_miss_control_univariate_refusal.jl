# Issue #1565: univariate `miss_control(response = ...)` must refuse with a
# bivariate-scope message, not the joint `mi(x)` / imputation wording.
using DRModels, StatsModels, Random, Test

@testset "univariate miss_control refusal names bivariate scope (#1565)" begin
    Random.seed!(26026)
    data = (
        y = [0.1, -0.3, NaN, 0.8, 0.2, -1.1, 0.4, 0.0, 0.6, -0.2],
        x = collect(range(-1.0, 1.0; length = 10)),
    )
    f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    err = try
        drm(f, Gaussian(); data = data, missing = miss_control(response = "fail"))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("miss_control", err.msg)
    @test occursin("bivariate", err.msg)
    @test !occursin("mi(x)", err.msg)
    @test !occursin("joint-model", err.msg)
end