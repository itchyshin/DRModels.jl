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

    @testset "a pair that adds only a correlation is refused" begin
        # (1 | g) + (0 + x | g) against (1 + x | g): the only extra parameter is the
        # correlation, whose null is interior, so the plain chi-square(1) is right.
        ind = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1)),
                  Gaussian(); data = d)
        corr = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)),
                   Gaussian(); data = d)
        err = try lrtest(ind, corr; mixture = true); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("mixture = true", err.msg)
        @test occursin("correlation", err.msg)
        plain = lrtest(ind, corr)               # the plain test still works and is the right one
        @test plain.dof == 1
        # a pair with different fixed effects and one added SD is refused by name too
        red_fx = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data = d)
        err = try lrtest(red_fx, full_re; mixture = true); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("fixed effects", err.msg)
    end

    @testset "a clearly negative statistic is refused" begin
        # `reduced` has the true grouping (1 | g); `full` swaps it for two unrelated groupings,
        # so the pair is not nested and the statistic is far below zero.
        Random.seed!(7)
        h = repeat(1:G, outer = m); k = mod.(3 .* h .+ g, 7)
        yb = 0.3 .+ 1.2 .* randn(G)[g] .+ 0.5 .* randn(nn)
        db = (; y = yb, h, k, g)
        red = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = db)
        notnested = drm(bf(@formula(y ~ 1 + (1 | h) + (1 | k)), @formula(sigma ~ 1)), Gaussian(); data = db)
        plain = lrtest(red, notnested)
        @test plain.statistic < -1.0
        err = try lrtest(red, notnested; mixture = true); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("not nested or a fit did not converge", err.msg)
        # the plain call keeps its old behaviour on the same pair: the statistic comes back as it is
        @test plain.pvalue == 1.0
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
