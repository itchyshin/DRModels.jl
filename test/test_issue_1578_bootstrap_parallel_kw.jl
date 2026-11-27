# DRModels.jl#1578 — drmTMB-style parallel= rejected with threads= guidance
using DRModels, Random, Test

function _argmsg(f)
    try
        f()
        error("expected an ArgumentError")
    catch err
        err isa ArgumentError || rethrow()
        return err.msg
    end
end

@testset "issue 1578: bootstrap rejects parallel= keyword" begin
    y = randn(60)
    data = (; y)
    fit = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data)
    rng = MersenneTwister(1)
    for (name, f) in (
        ("bootstrap_result", () -> bootstrap_result(fit; data, B = 2, rng, parallel = false)),
        ("bootstrap_ci", () -> bootstrap_ci(fit; data, B = 2, rng, parallel = true)),
    )
        msg = _argmsg(f)
        @test occursin("parallel", msg)
        @test occursin("threads", msg)
        @test occursin(name, msg)
    end
    @test bootstrap_result(fit; data, B = 2, rng, threads = false) !== nothing
end