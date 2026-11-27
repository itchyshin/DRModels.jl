# Issue #1613: drm_listwise(f, Poisson(); data = ...) must not surface a raw MethodError.
using DRModels, Test

@testset "drm_listwise refuses drm-style family + data= keyword (#1613)" begin
    f = bf(@formula(y ~ x))
    d = (; x = randn(5), y = fill(missing, 5))

    err = try
        drm_listwise(f, Poisson(); data = d)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("drm_listwise", msg)
    @test occursin("Poisson()", msg)
    @test occursin("positional", msg) || occursin("second", msg)
    @test occursin("drm(formula, family()", msg)
    @test !occursin("unsupported keyword", msg)

    err2 = try
        drm_listwise(f, Poisson())
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("Poisson()", sprint(showerror, err2))
end