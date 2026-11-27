# Issue #1610 — string numeric codes on a Poisson response used to reach
# `Float64(::String)` during design construction (cursor-cloud-1127fx).
using DRModels
using Random, StatsModels, Test

@testset "Poisson(): string numeric response column (issue #1610)" begin
    Random.seed!(88)
    data = (; x = randn(5), y = ["0", "1", "2", "1", "0"])
    err = try
        drm(bf(@formula(y ~ x)), Poisson(); data = data, control = DRMControl(se = false))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test startswith(msg, "ArgumentError: drm:")
    @test occursin("response `y`", msg)
    @test occursin("numeric", msg)
    @test occursin("Int", msg)
    @test occursin("Float64", msg)

    err2 = try
        DRModels._coerce_response_column(data.y; name = "y")
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test startswith(sprint(showerror, err2), "ArgumentError: drm:")
end