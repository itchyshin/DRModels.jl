# vc() and re_sd() on a structured fit whose only SD parameter is an intercept-only `:sd` block.
# `animal(1 | g)`, `relmat(1 | g)` and `spatial(1 | g)` on a Binomial response, and `spatial(1 | g)`
# on a Beta response, store the plain log SD in a block named `:sd`. That block has one coefficient,
# so it is one SD per grouping, not an SD regression. These accessors used to refuse it with a
# message about `sd(group) ~ ...`.
#
# The Binomial animal case is pinned to drmTMB 0.8.1 (data, script and recorded output in
# test/data/vc_structured_sd_intercept/): logLik -281.1827097, animal SD 0.1649602.
using DRModels, Test, DelimitedFiles, LinearAlgebra, Logging

@testset "vc / re_sd: intercept-only structured SD" begin
    dir = joinpath(@__DIR__, "data", "vc_structured_sd_intercept")
    raw, _ = readdlm(joinpath(dir, "vcA.csv"), ','; header = true)
    s, fl, x = Float64.(raw[:, 1]), Float64.(raw[:, 2]), Float64.(raw[:, 3])
    g = String.(raw[:, 4])
    G = 20; ids = ["a" * string(k) for k in 1:G]
    A = [exp(-abs(i - j) / 3) for i in 1:G, j in 1:G]
    data = (; s, fl, x, g)
    quiet(f) = with_logger(f, NullLogger())

    @testset "Binomial animal(1 | g) against drmTMB" begin
        f = quiet(() -> drm(bf(@formula(cbind(s, fl) ~ x + animal(1 | g))), Binomial(); data, A, ids))
        @test isapprox(loglikelihood(f), -281.1827097; atol = 1e-4)
        @test length(coef(f, :sd)) == 1
        sd = re_sd(f)
        @test collect(keys(sd)) == [:g]
        @test isapprox(sd[:g], 0.1649602; atol = 2e-6)
        V = vc(f)
        @test collect(keys(V)) == [:g]
        @test size(V[:g]) == (1, 1)
        @test isapprox(V[:g][1, 1], 0.1649602^2; rtol = 2e-5)
    end

    @testset "Binomial relmat and spatial give the SD the fit used" begin
        fr = quiet(() -> drm(bf(@formula(cbind(s, fl) ~ x + relmat(1 | g))), Binomial(); data, K = A))
        coords = reshape(Float64.(1:G), G, 1)
        fs = quiet(() -> drm(bf(@formula(cbind(s, fl) ~ x + spatial(1 | g))), Binomial(); data, coords))
        for f in (fr, fs)
            @test length(coef(f, :sd)) == 1
            @test re_sd(f)[:g] ≈ exp(only(coef(f, :sd)))
            @test vc(f)[:g][1, 1] ≈ exp(only(coef(f, :sd)))^2
        end
    end

    @testset "Beta spatial(1 | g)" begin
        h(i) = mod(sin(12.9898i) * 43758.5453, 1)
        y = [clamp(1 / (1 + exp(-(0.2 + 0.4x[i]))) * (0.7 + 0.6h(8000 + i)), 0.02, 0.98) for i in eachindex(x)]
        coords = reshape(Float64.(1:G), G, 1)
        f = quiet(() -> drm(bf(@formula(y ~ x + spatial(1 | g)), @formula(sigma ~ 1)), Beta();
                            data = (; y, x, g), coords))
        @test length(coef(f, :sd)) == 1
        @test re_sd(f)[:g] ≈ exp(only(coef(f, :sd)))
        @test vc(f)[:g][1, 1] ≈ exp(only(coef(f, :sd)))^2
    end

    @testset "a real SD regression is still refused" begin
        z = [i <= G ÷ 2 ? 0.0 : 1.0 for i in 1:G]
        zrow = [z[findfirst(==(gi), ids)] for gi in g]
        d2 = (; s, fl, x, g, z = zrow)
        f = quiet(() -> drm(bf(@formula(cbind(s, fl) ~ x + animal(1 | g)), @formula(sd(g, animal) ~ z)),
                            Binomial(); data = d2, A, ids))
        @test length(coef(f, :sd)) == 2
        @test_throws ArgumentError re_sd(f)
        @test_throws ArgumentError vc(f)
    end
end
