# Refs #997: `_marginal_simulator_build` returns `nothing` for several supported
# or mis-specified fits, and `bootstrap_result` then silently uses the
# CONDITIONAL `simulate` (random effects at 0; the #459 variance collapse).
# Only the `meta_V` + RE route throws. These tests document the current silent
# fallback; the desired throw/warn is @test_broken. No behaviour change.
using DRModels
using Test, Random

_threw(f) = try; f(); false catch e; e isa ArgumentError; end

@testset "marginal simulator silent fallback (#997)" begin
    rng = MersenneTwister(1)
    G1, G2, n = 10, 6, 120
    g1 = repeat(1:G1, inner = 12); g2 = repeat(1:G2, outer = 20)
    x = randn(rng, n)
    y = 0.2 .+ 0.3 .* x .+ randn(rng, G1)[g1] .+ 0.5 .* randn(rng, G2)[g2] .+ randn(rng, n)
    data = (; y, x, g1, g2)
    sd = DRModels._marginal_simulator_build
    ms = DRModels._marginal_simulator

    @testset "supported single (1 | g): a simulator is built" begin
        f = drm(bf(@formula(y ~ x + (1 | g1)), @formula(sigma ~ 1)), Gaussian(); data)
        @test sd(f, data) isa Function
        # Missing grouping column -> silent `nothing` (conditional fallback).
        @test sd(f, (; y, x)) === nothing
        @test ms(f, (; y, x)) === nothing
        @test_broken _threw(() -> ms(f, (; y, x)))
    end

    @testset "crossed ordinary bars: a simulator is now built" begin
        # Was a silent `nothing` (conditional fallback). Several ordinary intercepts on
        # distinct groups now get a fresh-draw simulator (`_ordinary_multi_marginal_simulator`).
        f = drm(bf(@formula(y ~ x + (1 | g1) + (1 | g2)), @formula(sigma ~ 1)),
                Gaussian(); data)
        @test sd(f, data) isa Function
        @test ms(f, data) isa Function
        # A repeated group is still not covered, so it stays a silent `nothing`.
        f2 = drm(bf(@formula(y ~ x + (1 | g1) + (0 + x | g1) + (1 | g2)), @formula(sigma ~ 1)),
                 Gaussian(); data)
        @test sd(f2, data) === nothing
    end

    @testset "correlated slope (1 + x | g): silent nothing" begin
        f = drm(bf(@formula(y ~ x + (1 + x | g1)), @formula(sigma ~ 1)), Gaussian(); data)
        @test sd(f, data) === nothing
        @test_broken _threw(() -> ms(f, data))
    end
end
