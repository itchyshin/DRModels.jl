# refs #1351 — bare FormulaTerm must be refused by name, not MethodError.
module TestIssue1351BareFormula

using DRModels
using Test

@testset "#1351 drm without bf()" begin
    x = sin.(1:40)
    y = 1.0 .+ 0.5 .* x
    err = try
        drm(@formula(y ~ x), Gaussian(); data = (; y, x))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("bf()", sprint(showerror, err))
end

end