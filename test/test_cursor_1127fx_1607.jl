using DRModels, Test

@testset "#1607: update without `data=` raises named ArgumentError" begin
    include(joinpath(@__DIR__, "parity", "loadfixture.jl"))
    data = load_data(joinpath(@__DIR__, "parity", "fixtures", "gaussian-locscale"))
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data)
    newform = bf(@formula(y ~ x), @formula(sigma ~ x))

    ex = try
        update(fit, newform)
        nothing
    catch e
        e
    end
    @test ex isa ArgumentError
    @test occursin("data", sprint(showerror, ex))
    reduced = update(fit, newform; data)
    @test length(coef(reduced)) >= length(coef(fit)) - 1
end