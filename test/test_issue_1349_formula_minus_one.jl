# refs #1349 — `@formula(y ~ x - 1)` must not silently keep an intercept.
module TestIssue1349FormulaMinusOne

using DRModels
using Test

@testset "#1349 refuse R-style x - 1" begin
    x = [2sin(i) for i in 1:60]
    z = [cos(3i) for i in 1:60]
    y = [3 + 0.5 * x[i] + 0.3cos(7i) for i in 1:60]
    d = (; y, x, z)
    err = try
        drm(bf(@formula(y ~ x - 1), @formula(sigma ~ 1)), Gaussian(); data = d)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("0 + x", sprint(showerror, err))
    fit0 = drm(bf(@formula(y ~ 0 + x), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test !any(==("(Intercept)"), fit0.coefnames[1].second)
end

end