# #1611: ordered CategoricalArray used as a Poisson response must not die in
# `Float64(::CategoricalValue)`; refuse with guidance instead.
using DRModels, CategoricalArrays, Random, Test

@testset "#1611: Poisson refuses ordered categorical response" begin
    Random.seed!(1)
    y = categorical([1, 2, 3, 1, 2, 3], ordered = true)
    data = (; x = randn(6), y)
    ex = try
        drm(bf(@formula(y ~ x)), Poisson(); data, control = DRMControl(se = false))
        nothing
    catch e
        e
    end
    @test ex isa ArgumentError
    msg = sprint(showerror, ex)
    @test occursin("categorical", lowercase(msg))
    @test occursin("Poisson", msg)
    @test !occursin("MethodError", msg)
end