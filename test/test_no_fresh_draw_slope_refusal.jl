using DRModels, Test, Random

# Gamma `(1 | g) + (0 + x | g)` has no fresh random-effect draw. Before this
# change `bootstrap_summary` fell back to a draw with the random effects at zero and returned a
# table (a variance-component interval that collapses); it now refuses by name. Binomial has its own
# test file (test_binomial_slope_no_fresh_draw.jl).
@testset "no fresh draw: Gamma (1 | g) + (0 + x | g) refuses bootstrap and default simulate" begin
    r = Random.Xoshiro(3); n = 240; G = 20
    g = string.("g", rand(r, 1:G, n)); x = randn(r, n)
    gi = parse.(Int, SubString.(g, 2)); bg = 0.5 .* randn(r, G); bs = 0.3 .* randn(r, G)
    eta = 0.3 .+ 0.4 .* x .+ bg[gi] .+ bs[gi] .* x
    yg = [exp(e) * (0.6 + 0.8 * rand(r)) for e in eta]
    dg = (; y = yg, x, g)
    fg = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g))), DRModels.Gamma(); data = dg)
    for (fit, d) in ((fg, dg),)
        err = try; bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("no fresh random-effect draw", sprint(showerror, err))
        err2 = try; simulate(fit; data = d, rng = Random.Xoshiro(1)); nothing; catch e; e; end
        @test err2 isa ArgumentError && occursin("no fresh random-effect draw", sprint(showerror, err2))
    end
    # the escape hatch named in the message works for Gamma
    @test length(simulate(fg; data = dg, re = :zero, rng = Random.Xoshiro(1))) == n
    # a single (1 | g) keeps its simulator and bootstrap
    f1 = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Gamma(); data = dg)
    @test size(bootstrap_summary(f1; data = dg, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(f1))
end

# The guard refuses EVERY Gamma fit with two or more
# ordinary random-effect terms that has no fresh-draw simulator, not only `(1 | g) + (0 + x | g)`.
# Probed on main: all of these shapes had no fresh draw (silent fallback to random effects at zero).
# Each refusal must name the shape actually fitted and offer the escape that works.
function _no_fresh_draw_data(seed)
    r = Random.Xoshiro(seed); n = 300
    gi = rand(r, 1:20, n); hi = rand(r, 1:10, n); si = rand(r, 1:3, n)
    x = randn(r, n); z = randn(r, n)
    bg = 0.4 .* randn(r, 20); bh = 0.4 .* randn(r, 10); bs = 0.3 .* randn(r, 20)
    eta = 0.3 .+ 0.4 .* x .+ bg[gi] .+ bh[hi] .+ bs[gi] .* z
    g = string.("g", gi); h = string.("h", hi); gh = string.("g", gi, "_", si)
    yg = [exp(e) * (0.6 + 0.8 * rand(r)) for e in eta]
    return (; x, z, g, h, gh, y = yg)
end

@testset "no fresh draw: every refused Gamma shape names itself and the escape works" begin
    d = _no_fresh_draw_data(11)
    shapes = [
        ("(1 | g) + (0 + z | g)", "an independent random slope with an intercept"),
        ("(1 | g) + (0 + z | g) + (1 | h)", "a mix of random intercepts and slopes"),
        ("(0 + x | g) + (0 + z | g)", "independent random slopes without an intercept"),
        ("(0 + x | g) + (0 + z | h)", "random slopes on different groups"),
    ]
    for fam in ("Gamma",), (terms, label) in shapes
        fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula(y ~ x + $terms))"))
        fit = drm(fo, DRModels.Gamma(); data = d)
        calls = (
            "bootstrap_summary" => () -> bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
            "bootstrap_result" => () -> bootstrap_result(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
            "default simulate" => () -> simulate(fit; data = d, rng = Random.Xoshiro(1)),
        )
        msg = ""
        for (cname, call) in calls
            err = try; call(); nothing; catch e; e; end
            @test err isa ArgumentError
            err isa ArgumentError || continue
            msg = sprint(showerror, err)
            @test occursin("no fresh random-effect draw", msg)
            @test occursin("for $fam() with $label ($terms)", msg)
            # a crossed or mixed fit must not be described as the slope-with-intercept shape
            label == "an independent random slope with an intercept" ||
                @test !occursin("an independent random slope with an intercept", msg)
            # slope-only shapes have no intercept and must never be said to have one
            occursin("slopes", label) && !occursin("with an intercept", label) &&
                @test !occursin("with an intercept", msg)
        end
        # the escape named in the message works for this fit
        @test occursin("call `simulate(fit; data = data, re = :zero)`", msg)
        @test length(simulate(fit; data = d, re = :zero, rng = Random.Xoshiro(1))) == length(d.x)
    end
end

@testset "no fresh draw: crossed and nested Gamma intercepts have a fresh draw (simulator for several random intercepts)" begin
    d = _no_fresh_draw_data(11)
    gv(y, grp) = (ks = unique(grp); m = [sum(y[grp .== k]) / count(==(k), grp) for k in ks];
                  sum(abs2, m .- sum(m) / length(m)) / (length(m) - 1))
    for terms in ("(1 | g) + (1 | h)", "(1 | g) + (1 | gh)")
        fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula(y ~ x + $terms))"))
        fit = drm(fo, DRModels.Gamma(); data = d)
        @test size(bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(fit))
        fresh = sum(gv(simulate(fit; data = d, rng = Random.Xoshiro(s)), d.g) for s in 1:40) / 40
        # between-group variance of the simulated group means against the data's (own bound, no R value)
        @test 0.5 < fresh / gv(d.y, d.g) < 2
    end
end

@testset "no fresh draw: shapes that keep a working simulator or were never in scope do not refuse" begin
    d = _no_fresh_draw_data(11)
    # a single (1 | g) keeps its fresh draw: default simulate and bootstrap work
    fg = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Gamma(); data = d)
    @test length(simulate(fg; data = d, rng = Random.Xoshiro(1))) == length(d.x)
    @test size(bootstrap_summary(fg; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(fg))
    # single-term Gamma slope shapes are covered by test_gamma_slope_no_fresh_draw.jl
    # Beta, Student and LogNormal with crossed or slope bars refuse too now (a maintainer decision);
    # see test_slope_bootstrap_refusal_more_families.jl
end
