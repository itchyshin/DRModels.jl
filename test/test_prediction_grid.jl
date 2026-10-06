# prediction_grid — the pure-data newdata builder that completes the parameter-
# prediction surface. It sweeps one or more predictors over supplied ranges
# (Cartesian product) while holding every other predictor at a reference value.
# Independent of DrmFit (operates on plain NamedTuples), but must compose with
# predict_parameters(fit, prediction_grid(...)).
using DRModels
using Test, Random
using Statistics: mean

@testset "prediction_grid" begin

    @testset "single swept predictor: range varies, others held" begin
        g = prediction_grid((; x = [0.0], z = [1.0, 2.0, 3.0]), x = -2:1.0:2)
        @test length(g.x) == 5
        @test g.x == collect(-2:1.0:2)
        # z held at mean([1,2,3]) == 2.0, broadcast to the product length.
        @test length(g.z) == 5
        @test all(==(mean([1.0, 2.0, 3.0])), g.z)
        @test all(==(2.0), g.z)
    end

    @testset "no kwargs → single reference row" begin
        g = prediction_grid((; x = [1.0, 3.0], w = "a"))
        @test length(g.x) == 1
        @test length(g.w) == 1
        @test g.x[1] == mean([1.0, 3.0])    # numeric array → mean
        @test g.w[1] == "a"                 # scalar held as-is
    end

    @testset "non-numeric reference array → first element" begin
        g = prediction_grid((; grp = ["a", "b", "c"]), x = 0.0:1.0:2.0)
        @test length(g.x) == 3
        @test all(==("a"), g.grp)           # first element of a non-numeric array
    end

    @testset "two swept predictors → full Cartesian product" begin
        xs = range(-1, 1; length = 4)
        zs = [10.0, 20.0, 30.0]
        g = prediction_grid((; x = [0.0], z = [0.0], w = [5.0, 7.0]), x = xs, z = zs)
        @test length(g.x) == length(xs) * length(zs)   # 4 * 3 == 12
        @test length(g.z) == length(xs) * length(zs)
        @test length(g.w) == length(xs) * length(zs)
        # w is held (not swept) → constant at mean([5,7]) == 6.0.
        @test all(==(6.0), g.w)

        # Every (x, z) combination is present exactly once.
        combos = Set((g.x[i], g.z[i]) for i in eachindex(g.x))
        expected = Set((xi, zi) for xi in xs for zi in zs)
        @test combos == expected
        @test length(combos) == length(xs) * length(zs)
    end

    @testset "swept predictor overrides held reference value" begin
        # x appears in both reference and kwargs → kwargs sweep wins.
        g = prediction_grid((; x = [99.0]), x = [1.0, 2.0])
        @test g.x == [1.0, 2.0]
    end

    @testset "composes with a Gaussian loc-scale fit" begin
        Random.seed!(20260604)
        n = 400
        x = randn(n)
        y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
        data = (; y, x)

        fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

        g = prediction_grid((; x = data.x), x = range(-1, 1; length = 7))
        @test length(g.x) == 7

        pp = predict_parameters(fit, g)
        @test Set(keys(pp)) == Set([:mu, :sigma])
        @test length(pp[:mu]) == 7
        @test length(pp[:sigma]) == 7
    end
end
