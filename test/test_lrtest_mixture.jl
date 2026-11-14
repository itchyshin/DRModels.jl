# lrtest(reduced, full; mixture = true): equal mixture of chi-square(0) and
# chi-square(1) when the pair differs by exactly one variance component.
using DRModels, Test, Random
import Distributions

@testset "lrtest mixture = true" begin
    Random.seed!(20261114)
    G = 20; m = 10; nn = G * m
    g = repeat(1:G, inner = m); x = randn(nn)
    b = 0.6 .* randn(G)
    y = 0.5 .- 0.4 .* x .+ b[g] .+ 0.7 .* randn(nn)
    d = (; y, x, g)
    full_re = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
    red_re  = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)

    @testset "one random intercept left out" begin
        plain = (@test_logs (:warn,) match_mode = :any lrtest(red_re, full_re))
        t = (@test_logs lrtest(red_re, full_re; mixture = true))   # no warning
        @test t.statistic == plain.statistic
        @test t.dof == 1
        @test t.pvalue_naive == plain.pvalue
        @test t.reference === :chibar_0_1
        @test t.statistic > 0
        @test t.pvalue ≈ 0.5 * Distributions.ccdf(Distributions.Chisq(1), t.statistic) rtol = 1e-12
        @test t.pvalue ≈ lrt_boundary(full_re, red_re; q = 1).pvalue rtol = 1e-12
        @test t.pvalue < plain.pvalue
        # the default is unchanged: same fields, plain chi-square
        @test keys(plain) == (:statistic, :dof, :pvalue)
    end

    @testset "second independent slope SD is one more variance" begin
        full2 = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1)),
                    Gaussian(); data = d)
        t = lrtest(full_re, full2; mixture = true)
        @test t.dof == 1
        @test t.pvalue ≈ 0.5 * Distributions.ccdf(Distributions.Chisq(1), max(t.statistic, 0)) rtol = 1e-12
    end

    @testset "refusals" begin
        # slope with its correlation: two SDs and a correlation against one SD
        corr = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)),
                   Gaussian(); data = d)
        err = try lrtest(full_re, corr; mixture = true); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("mixture = true", err.msg)
        @test occursin("different mixture", err.msg)
        # no variance component added: fixed effects only
        fx = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data = d)
        fx2 = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)
        err = try lrtest(fx, fx2; mixture = true); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("plain", err.msg)
        # the plain call still works on the refused pair
        @test lrtest(full_re, corr).dof == dof(corr) - dof(full_re)
    end

    @testset "Poisson inside NegBinomial2" begin
        Random.seed!(11)
        xx = randn(120)
        yy = Float64.(rand.(Distributions.Poisson.(exp.(0.4 .+ 0.3 .* xx))))
        dd = (; y = yy, x = xx)
        pois = drm(bf(@formula(y ~ 1 + x)), DRModels.Poisson(); data = dd, se = false)
        nb = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2();
                 data = dd, se = false)
        t = lrtest(pois, nb; mixture = true)
        @test t.pvalue ≈ lrt_boundary(nb, pois; q = 1).pvalue rtol = 1e-12
    end
end
