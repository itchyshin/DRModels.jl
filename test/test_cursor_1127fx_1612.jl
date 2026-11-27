# Issue #1612 — Poisson mean `y ~ 0` with no covariates must refuse, not BoundsError.
using DRModels
using Test, Random

@testset "Poisson y ~ 0 empty mean design (#1612)" begin
    rng = MersenneTwister(42)
    y = rand(rng, 0:2, 8)
    data = (; y)

    @test_throws ArgumentError drm(
        bf(@formula(y ~ 0)), Poisson();
        data = data, control = DRMControl(se = false))

    msg = try
        drm(bf(@formula(y ~ 0)), Poisson(); data = data, control = DRMControl(se = false))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("Poisson()", msg)
    @test occursin("no estimable fixed-effect", msg)
    @test occursin("y ~ 0", msg)

    # Unrelated routes stay usable: intercept-only and random intercept without FE columns.
    @test drm(bf(@formula(y ~ 1)), Poisson(); data = data, control = DRMControl(se = false)) isa DrmFit
    g = repeat(1:4, inner = 2)
    @test drm(bf(@formula(y ~ 0 + (1 | g))), Poisson(); data = (y = y, g = g),
            control = DRMControl(se = false)) isa DrmFit
end