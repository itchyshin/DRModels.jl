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

# The refusal margin is `-1e-6 * max(1, |loglik|)`. A nested pair whose group variance is estimated at
# zero gives a statistic of optimiser-noise size (below 1e-8 here), which must be accepted at any data size;
# a pair that is not nested must be refused at any data size. The same two checks were also run with
# 500000 rows (statistic 1.9e-6 for the boundary pair, -5.8e5 for the non-nested pair): not repeated here
# because the fits take about half a minute each.
@testset "negative-statistic margin at 50 and 5000 rows" begin
    for (n, G) in ((50, 10), (5000, 100))
        rng = Random.MersenneTwister(3); m = n ÷ G
        g = repeat(1:G, inner = m); h = repeat(1:G, outer = m); k = mod.(3 .* h .+ g, 7)
        x = repeat(collect(range(-1.5, 1.5; length = m)), outer = G)
        e = 0.7 .* randn(rng, n)
        for j in 1:G
            r = ((j - 1) * m + 1):(j * m)
            e[r] .-= sum(e[r]) / m                      # every group mean is zero: the group variance is 0
        end
        d0 = (; y = 0.5 .- 0.4 .* x .+ e, x, g)
        red0 = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d0)
        full0 = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d0)
        t = lrtest(red0, full0; mixture = true)
        @test abs(t.statistic) < 1e-8
        @test t.reference === :chibar_0_1
        # not nested: y follows the grouping h, the full model has two unrelated groupings
        yb = 0.3 .+ 1.2 .* randn(rng, G)[h] .+ 0.5 .* randn(rng, n)
        d1 = (; y = yb, h, k, g)
        red1 = drm(bf(@formula(y ~ 1 + (1 | h)), @formula(sigma ~ 1)), Gaussian(); data = d1)
        non = drm(bf(@formula(y ~ 1 + (1 | g) + (1 | k)), @formula(sigma ~ 1)), Gaussian(); data = d1)
        @test lrtest(red1, non).statistic < -1.0
        err = try lrtest(red1, non; mixture = true); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("not nested or a fit did not converge", err.msg)
    end
end

# A slope on the log standard deviation is an interior null. Counting every column of
# the sd block as a variance accepted these pairs and halved the p-value.
@testset "a log-SD slope is not a variance component" begin
    Random.seed!(23)
    G = 16; m = 6; nn = G * m
    g = repeat(1:G, inner = m)
    z = repeat(randn(G), inner = m)
    x = randn(nn)
    b = exp.(-0.2 .+ 0.35 .* z[1:m:end]) .* randn(G)
    y = 0.3 .+ 0.25 .* x .+ b[g] .+ 0.5 .* randn(nn)
    d = (; y, x, z, g)
    sd1 = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ 1)),
              Gaussian(); data = d)
    sdz = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ z)),
              Gaussian(); data = d)
    re = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test dof(sdz) - dof(sd1) == 1
    @test dof(sdz) - dof(re) == 1
    for (reduced, full) in ((sd1, sdz), (re, sdz))
        err = try lrtest(reduced, full; mixture = true); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("mixture = true", err.msg)
        @test occursin("interior", err.msg)
        @test occursin("slope", err.msg)
        plain = lrtest(reduced, full)
        @test plain.dof == 1
        @test plain.pvalue ≈ Distributions.ccdf(Distributions.Chisq(1), max(plain.statistic, 0)) rtol = 1e-12
    end
end

# Counts differ by one SD, but the labels do not: `(1 | g)` is not a component of
# `(1 | h) + (1 | k)`. The mixture applied whenever the statistic was positive.
@testset "swapped groupings are not one added variance" begin
    Random.seed!(29)
    G = 18; m = 8; nn = G * m
    g = repeat(1:G, inner = m)
    h = repeat(1:G, outer = m)
    k = mod.(h .+ 5, G) .+ 1
    y = 0.2 .+ 1.6 .* randn(G)[h] .+ 1.3 .* randn(G)[k] .+ 0.3 .* randn(nn)
    d = (; y, g, h, k)
    red = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
    full = drm(bf(@formula(y ~ 1 + (1 | h) + (1 | k)), @formula(sigma ~ 1)), Gaussian(); data = d)
    plain = lrtest(red, full)
    @test plain.dof == 1
    @test plain.statistic > 1.0
    err = try lrtest(red, full; mixture = true); nothing catch e; e end
    @test err isa ArgumentError
    @test occursin("mixture = true", err.msg)
    @test occursin("label", err.msg)
    # The example in the message always names `(1 | g)` and `(1 | h)`, so the
    # check has to be the labels the guard actually printed.
    @test occursin("[resd:g]", err.msg)
    @test occursin("[resd:h, resd:k]", err.msg)
end

# A one-group `sd(g) ~ 1` is labelled `sd:(Intercept)`. Two groups, or a plain
# `(1 | h)` once any sd formula is present, are `sd:g: (Intercept)` and
# `sd:h: (Intercept)`. A plain random intercept is `resd:g`. Those spellings are
# one component. Seed 41, G = 20, m = 8.
@testset "sd-block labels and plain random intercepts are the same group" begin
    Random.seed!(41)
    G = 20; m = 8; nn = G * m
    g = repeat(1:G, inner = m)
    h = repeat(1:G, outer = m)
    x = randn(nn)
    y = 0.2 .+ 0.4 .* x .+ randn(G)[g] .+ 0.7 .* randn(G)[h] .+ 0.5 .* randn(nn)
    d = (; y, x, g, h)
    red_sd = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ 1)),
                 Gaussian(); data = d)
    full_sd = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1),
                   @formula(sd(g) ~ 1)),
                  Gaussian(); data = d)
    # (a) sd(g) ~ 1 on both sides; full adds (1 | h)
    t = lrtest(red_sd, full_sd; mixture = true)
    @test t.dof == 1
    @test t.reference === :chibar_0_1
    @test t.pvalue ≈ 0.5 * Distributions.ccdf(Distributions.Chisq(1), max(t.statistic, 0)) rtol = 1e-12
    # (c) plain (1 | g), no sd formula, against the same full model
    red_re = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
    t = lrtest(red_re, full_sd; mixture = true)
    @test t.dof == 1
    @test t.reference === :chibar_0_1
    @test t.pvalue ≈ 0.5 * Distributions.ccdf(Distributions.Chisq(1), max(t.statistic, 0)) rtol = 1e-12
end

# Plain `phylo(1 | species)` is `resd:species`. The same intercept beside another
# group, with `sd(species, phylogenetic) ~ 1`, is `sd_phylo:(Intercept)`.
@testset "a phylogenetic intercept keeps its group under sd_phylo" begin
    Random.seed!(41)
    phy = DRModels.augmented_phy("((t1:1,t2:1):1,(t3:1,t4:1):1);")
    tips = String.(phy.leaf_names)
    m = 4
    species = repeat(tips, inner = m)
    ng = length(tips)
    g = repeat(1:ng, outer = m)
    nn = length(species)
    x = randn(nn)
    tip_eff = Dict(tips[i] => 0.8 * randn() for i in eachindex(tips))
    y = 0.3 .+ 0.25 .* x .+ [tip_eff[s] for s in species] .+ 0.6 .* randn(ng)[g] .+
        0.4 .* randn(nn)
    d = (; y, x, species, g)
    red = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
              Gaussian(); data = d, tree = phy)
    full = drm(bf(@formula(y ~ x + (1 | g) + phylo(1 | species)),
                  @formula(sigma ~ 1), @formula(sd(species, phylogenetic) ~ 1)),
               Gaussian(); data = d, tree = phy)
    t = lrtest(red, full; mixture = true)
    @test t.dof == 1
    @test t.reference === :chibar_0_1
    @test t.pvalue ≈ 0.5 * Distributions.ccdf(Distributions.Chisq(1), max(t.statistic, 0)) rtol = 1e-12
end

# `phylo(1 | sp)` and `(1 | sp)` are both stored as `resd:sp`. They are not the
# same component. 4-tip tree, m = 6, seed 41. Each statistic is positive, so
# the refusal is the label guard.
@testset "a phylogenetic intercept is not an iid intercept on the same group" begin
    Random.seed!(41)
    phy = DRModels.augmented_phy("((t1:1,t2:1):1,(t3:1,t4:1):1);")
    tips = String.(phy.leaf_names)
    m = 6
    sp = repeat(tips, inner = m)
    ng = length(tips)
    g = repeat(1:ng, outer = m)
    nn = length(sp)
    x = randn(nn)
    tip_eff = Dict(tips[i] => 0.9 * randn() for i in eachindex(tips))
    y = 0.3 .+ 0.25 .* x .+ [tip_eff[s] for s in sp] .+ 0.8 .* randn(ng)[g] .+
        0.35 .* randn(nn)
    d = (; y, x, sp, g)
    sig = @formula(sigma ~ 1)
    phylo_only = drm(bf(@formula(y ~ x + phylo(1 | sp)), sig), Gaussian(); data = d, tree = phy)
    iid_only = drm(bf(@formula(y ~ x + (1 | sp)), sig), Gaussian(); data = d)
    iid_both = drm(bf(@formula(y ~ x + (1 | sp) + (1 | g)), sig), Gaussian(); data = d)
    phylo_g = drm(bf(@formula(y ~ x + phylo(1 | sp) + (1 | g)), sig),
                  Gaussian(); data = d, tree = phy)
    function refused(reduced, full)
        err = try
            lrtest(reduced, full; mixture = true)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("mixture = true", err.msg)
        @test occursin("label", err.msg)
        @test occursin("not that pair", err.msg)
        return err
    end
    # (1) phylo(1 | sp) against (1 | sp) + (1 | g)
    err = refused(phylo_only, iid_both)
    @test occursin("[resd:sp]", err.msg)
    @test occursin("resd:g", err.msg)
    # (2) (1 | sp) against phylo(1 | sp) + (1 | g)
    err = refused(iid_only, phylo_g)
    @test occursin("[resd:sp]", err.msg)
    @test occursin("resd:g", err.msg)
end

# `spatial(1 | g)` is also stored as `resd:g`, the same string as `(1 | g)`.
# The public Gaussian route fixes the range, so the fit has one SD and no
# `:range` block, and `spatial(1 | g)` against `(1 | g) + (1 | h)` would pass
# the one-SD count. `spatial(1 | g)` cannot be combined with an ordinary bar
# (the router refuses it), so the mirror pair does not fit. `animal(1 | g)` and
# `relmat(1 | g)` use the same bare-group `resd` label and the same `g_iid`
# collision label when an ordinary intercept shares the grouping; the structure
# key covers them on that metadata.
@testset "a spatial intercept is not an iid intercept on the same group" begin
    Random.seed!(41)
    G = 4
    m = 6
    nn = G * m
    g = repeat(1:G, inner = m)
    h = repeat(1:G, outer = m)
    x = randn(nn)
    y = 0.2 .+ 0.3 .* x .+ (0.9 .* randn(G))[g] .+ (0.7 .* randn(G))[h] .+ 0.35 .* randn(nn)
    d = (; y, x, g, h)
    coords = Dict(i => [Float64(i), 0.0] for i in 1:G)
    sig = @formula(sigma ~ 1)
    spatial_only = drm(bf(@formula(y ~ x + spatial(1 | g)), sig), Gaussian();
                       data = d, coords = coords)
    iid_both = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), sig), Gaussian(); data = d)
    err = try
        lrtest(spatial_only, iid_both; mixture = true)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("mixture = true", err.msg)
    @test occursin("label", err.msg)
    @test occursin("not that pair", err.msg)
    @test occursin("[resd:g]", err.msg)
    @test occursin("resd:h", err.msg)
end
