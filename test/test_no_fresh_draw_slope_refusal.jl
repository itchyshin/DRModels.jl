using DRModels, Test, Random

# P11c: Gamma and Binomial `(1 | g) + (0 + x | g)` have no fresh random-effect draw. Before this
# change `bootstrap_summary` fell back to a draw with the random effects at zero and returned a
# table (a variance-component interval that collapses); it now refuses by name.
@testset "no fresh draw: Gamma and Binomial (1 | g) + (0 + x | g) refuse bootstrap and default simulate" begin
    r = Random.Xoshiro(3); n = 240; G = 20
    g = string.("g", rand(r, 1:G, n)); x = randn(r, n)
    gi = parse.(Int, SubString.(g, 2)); bg = 0.5 .* randn(r, G); bs = 0.3 .* randn(r, G)
    eta = 0.3 .+ 0.4 .* x .+ bg[gi] .+ bs[gi] .* x
    yg = [exp(e) * (0.6 + 0.8 * rand(r)) for e in eta]
    p = 1 ./ (1 .+ exp.(-eta)); s = Float64[sum(rand(r) < pi for _ in 1:10) for pi in p]
    dg = (; y = yg, x, g); db = (; s, f = 10.0 .- s, x, g)
    fg = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g))), DRModels.Gamma(); data = dg)
    fb = drm(bf(@formula(cbind(s, f) ~ x + (1 | g) + (0 + x | g))), Binomial(); data = db)
    for (fit, d) in ((fg, dg), (fb, db))
        err = try; bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("no fresh random-effect draw", sprint(showerror, err))
        err2 = try; simulate(fit; data = d, rng = Random.Xoshiro(1)); nothing; catch e; e; end
        @test err2 isa ArgumentError && occursin("no fresh random-effect draw", sprint(showerror, err2))
    end
    # the escape hatch named in the message works for Gamma (Binomial needs stored trials, not on main)
    @test length(simulate(fg; data = dg, re = :zero, rng = Random.Xoshiro(1))) == n
    # a single (1 | g) keeps its simulator and bootstrap
    f1 = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Gamma(); data = dg)
    @test size(bootstrap_summary(f1; data = dg, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(f1))
end

# P11c follow-up (review section 3): the guard refuses EVERY Gamma and Binomial fit with two or more
# ordinary random-effect terms that has no fresh-draw simulator, not only `(1 | g) + (0 + x | g)`.
# Probed on main: all of these shapes had no fresh draw (silent fallback to random effects at zero, or
# a failed refit / "does not carry trials" error). Each refusal must name the shape actually fitted
# and offer only an escape that works for the family.
function _p11c_data(seed)
    r = Random.Xoshiro(seed); n = 300
    gi = rand(r, 1:20, n); hi = rand(r, 1:10, n); si = rand(r, 1:3, n)
    x = randn(r, n); z = randn(r, n)
    bg = 0.4 .* randn(r, 20); bh = 0.4 .* randn(r, 10); bs = 0.3 .* randn(r, 20)
    eta = 0.3 .+ 0.4 .* x .+ bg[gi] .+ bh[hi] .+ bs[gi] .* z
    g = string.("g", gi); h = string.("h", hi); gh = string.("g", gi, "_", si)
    yg = [exp(e) * (0.6 + 0.8 * rand(r)) for e in eta]
    p = 1 ./ (1 .+ exp.(-eta)); s = Float64[sum(rand(r) < pi for _ in 1:10) for pi in p]
    return (; x, z, g, h, gh, y = yg, s, f = 10.0 .- s)
end

@testset "no fresh draw: every refused Gamma and Binomial shape names itself; escapes work or are not offered" begin
    d = _p11c_data(11)
    shapes = [
        ("(1 | g) + (0 + z | g)", "an independent random slope with an intercept"),
        ("(1 | g) + (1 | h)", "crossed or nested random intercepts"),
        ("(1 | g) + (1 | gh)", "crossed or nested random intercepts"),
        ("(1 | g) + (0 + z | g) + (1 | h)", "a mix of random intercepts and slopes"),
        ("(0 + x | g) + (0 + z | g)", "independent random slopes without an intercept"),
        ("(0 + x | g) + (0 + z | h)", "random slopes on different groups"),
    ]
    for fam in ("Gamma", "Binomial"), (terms, label) in shapes
        lhs = fam == "Gamma" ? "y" : "cbind(s, f)"
        famobj = fam == "Gamma" ? DRModels.Gamma() : Binomial()
        fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula($lhs ~ x + $terms))"))
        fit = drm(fo, famobj; data = d)
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
        # the escape in the message is offered exactly when it works for this fit
        works = try; length(simulate(fit; data = d, re = :zero, rng = Random.Xoshiro(1))) == length(d.x); catch; false; end
        @test occursin("call `simulate(fit; data = data, re = :zero)`", msg) == works
        fam == "Gamma" && @test works
        fam == "Binomial" && !works && @test occursin("no draw is available yet", msg)
    end
end

@testset "no fresh draw: shapes that keep a working simulator or were never in scope do not refuse" begin
    d = _p11c_data(11)
    # a single (1 | g) keeps its fresh draw for both families: default simulate and bootstrap work
    fg = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Gamma(); data = d)
    fb = drm(bf(@formula(cbind(s, f) ~ x + (1 | g))), Binomial(); data = d)
    for fit in (fg, fb)
        @test length(simulate(fit; data = d, rng = Random.Xoshiro(1))) == length(d.x)
        @test size(bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(fit))
    end
    # single-term Gamma slope shapes are covered by test_gamma_slope_no_fresh_draw.jl (decision f)
    # other families with crossed or independent-slope bars are untouched by the Gamma/Binomial guard
    fn = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), LogNormal(); data = d)
    @test size(bootstrap_summary(fn; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(fn))
end
