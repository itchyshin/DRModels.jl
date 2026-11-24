# refs #1355 — non-default factor contrasts are not supported; refusal names treatment coding.
import StatsModels: EffectsCoding
using DRModels, Test

@testset "#1355 contrasts keyword is refused with a treatment-contrast note" begin
    g = repeat(["a", "b", "c"], 20)
    y = randn(60)
    msg = try
        drm(bf(@formula(y ~ g)), Gaussian(); data = (; y, g),
            contrasts = Dict(:g => EffectsCoding()))
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow(e)
    end
    @test occursin("`contrasts`", msg)
    @test occursin("treatment contrasts", msg)
    @test occursin("Formula grammar", msg)
end