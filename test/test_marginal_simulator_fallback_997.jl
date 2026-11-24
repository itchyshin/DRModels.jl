# Refs #997: `_marginal_simulator_build` returns `nothing` for several supported
# or mis-specified fits, and `bootstrap_result` then silently uses the
# CONDITIONAL `simulate` (random effects at 0; the #459 variance collapse).
# Only the `meta_V` + RE route throws. These tests document the current silent
# fallback; the desired throw is @test_broken. The fallback now emits one @warn
# (message only; the returned simulator and all results are unchanged).
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
        @test @test_logs((:warn, r"could not build the marginal simulator"), ms(f, (; y, x))) === nothing
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

    @testset "correlated slope (1 + x | g): no simulator, now refused by name (#1387)" begin
        f = drm(bf(@formula(y ~ x + (1 + x | g1)), @formula(sigma ~ 1)), Gaussian(); data)
        @test sd(f, data) === nothing
        @test _threw(() -> ms(f, data))
    end

    @testset "meta_V-only fit: no warning (the conditional draw is the marginal)" begin
        v = fill(0.1, n)
        fm = drm(bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1)), Gaussian(); data = (; y, x, v))
        @test sd(fm, (; y, x, v)) === nothing
        @test DRModels._mu_has_random_effect(fm) == false
        @test_logs min_level = Base.CoreLogging.Warn ms(fm, (; y, x, v))
    end

    @testset "random-effect shapes the mean route parses only with extra flags still warn" begin
        # `_mu_has_random_effect` once parsed with `allow_phylo_slope` only, so these shapes threw an
        # ArgumentError, the catch returned false, and the warning was skipped. Reuse a cheap fit and
        # swap in the formula (the predicate reads only `fit.formula`, and it alone gates the warning).
        base = drm(bf(@formula(y ~ x + (1 | g1)), @formula(sigma ~ 1)), Gaussian(); data)
        with_formula(f) = DRModels.DrmFit(
            (name === :formula ? f : getfield(base, name) for name in fieldnames(DRModels.DrmFit))...)
        for (label, mu) in (
                "animal slope" => @formula(y ~ x + animal(1 + x | g1)),
                "relmat slope" => @formula(y ~ x + relmat(1 + x | g1)),
                "spatial slope" => @formula(y ~ x + spatial(1 + x | g1)),
                "temporal term" => @formula(y ~ x + temporal(1 | g1, g2, ar1)),
                "labelled term" => @formula(y ~ x + phylo(1 + x | p | g1)))
            @testset "$label" begin
                ff = with_formula(bf(mu, @formula(sigma ~ 1)))
                @test DRModels._mu_has_random_effect(ff) == true
            end
        end
    end

    @testset "valid fit with a simulator is silent" begin
        f = drm(bf(@formula(y ~ x + (1 | g1)), @formula(sigma ~ 1)), Gaussian(); data)
        sim = @test_logs min_level = Base.CoreLogging.Warn ms(f, data)
        @test sim isa Function
    end
end
