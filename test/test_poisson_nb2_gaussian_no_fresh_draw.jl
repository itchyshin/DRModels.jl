using DRModels, Test, Random

# #1387. Poisson and NB2 fits with a random slope (`(0 + z | g)`, `(1 + z | g)`) or an intercept plus an
# independent slope, and Gaussian fits with `(1 + z | g)` or an intercept plus an independent slope, have
# no fresh random-effect draw. `bootstrap_summary`, `bootstrap_result`, `bootstrap_ci` and the default
# `simulate(fit; data = data)` used to fall back to a draw with every random effect at zero (a warning for
# the bootstrap only), which makes the fixed-effect intervals far too narrow and the variance-component
# intervals unusable. They now refuse by name, with the same message form as the Gamma and Binomial refusals.
function _pn_data(seed)
    r = Random.Xoshiro(seed); n = 300; G = 20
    gi = rand(r, 1:G, n); hi = rand(r, 1:8, n); x = randn(r, n); z = randn(r, n)
    b0 = 0.4 .* randn(r, G); b1 = 0.3 .* randn(r, G); c0 = 0.3 .* randn(r, 8)
    eta = 0.3 .+ 0.3 .* x .+ b0[gi] .+ b1[gi] .* z .+ c0[hi]
    pois(l) = (L = exp(-l); k = 0; p = 1.0; while true; p *= rand(r); p <= L && return k; k += 1; end)
    yp = Float64[pois(exp(e)) for e in eta]
    # NB2 counts: Poisson with a gamma-like multiplier
    yn = Float64[pois(exp(e) * (0.5 + rand(r))) for e in eta]
    yg = eta .+ 0.5 .* randn(r, n)
    return (; x, z, g = string.("g", gi), h = string.("h", hi), yp, yn, yg)
end

const _PN_FAMILIES = [("Poisson", Poisson(), :yp, ""), ("NegBinomial2", NegBinomial2(), :yn, ""),
                      ("Gaussian", Gaussian(), :yg, "")]

@testset "Poisson, NB2, Gaussian slopes: bootstrap and default simulate refuse by name" begin
    d = _pn_data(7)
    cases = [
        ("(0 + z | g)", "a random slope without an intercept", ("Poisson", "NegBinomial2")),
        ("(1 + z | g)", "a correlated random intercept and slope", ("Poisson", "NegBinomial2", "Gaussian")),
        ("(1 | g) + (0 + z | g)", "an independent random slope with an intercept",
            ("Poisson", "NegBinomial2", "Gaussian")),
    ]
    for (fname, fam, resp) in ((f, m, r) for (f, m, r, _) in _PN_FAMILIES), (terms, label, fams) in cases
        fname in fams || continue
        @testset "$fname $terms" begin
            sig = fname == "Poisson" ? "" : ", @formula(sigma ~ 1)"
            fo = "bf(@formula($resp ~ x + $terms)$sig)"
            fit = drm(Core.eval(@__MODULE__, Meta.parse(fo)), fam; data = d)
            calls = (
                "bootstrap_summary" => () -> bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
                "bootstrap_result" => () -> bootstrap_result(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
                "bootstrap_ci" => () -> bootstrap_ci(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
                "default simulate" => () -> simulate(fit; data = d, rng = Random.Xoshiro(1)),
            )
            for (cname, call) in calls
                err = try; call(); nothing; catch e; e; end
                @test err isa ArgumentError
                err isa ArgumentError || continue
                msg = sprint(showerror, err)
                @test occursin("no fresh random-effect draw", msg)
                @test occursin("for $fname() with $label ($terms)", msg)
                @test occursin("random effects set to zero", msg)
                @test occursin("call `simulate(fit; data = data, re = :zero)`", msg)
            end
            # the escape named in the message works: a conditional draw with the random effects at zero
            @test length(simulate(fit; data = d, re = :zero, rng = Random.Xoshiro(1))) == length(d.x)
        end
    end
end

@testset "Poisson, NB2, Gaussian refusal leaves the working shapes alone" begin
    d = _pn_data(7)
    for (fname, fam, resp) in ((f, m, r) for (f, m, r, _) in _PN_FAMILIES)
        sig = fname == "Poisson" ? "" : ", @formula(sigma ~ 1)"
        for terms in ("(1 | g)", "(1 | g) + (1 | h)")
            fit = drm(Core.eval(@__MODULE__, Meta.parse("bf(@formula($resp ~ x + $terms)$sig)")), fam; data = d)
            @test length(simulate(fit; data = d, rng = Random.Xoshiro(1))) == length(d.x)
        end
    end
    # a Gaussian `(0 + z | g)` keeps its fresh draw and its bootstrap
    fg = drm(bf(@formula(yg ~ x + (0 + z | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
    @test length(simulate(fg; data = d, rng = Random.Xoshiro(1))) == length(d.x)
    @test size(bootstrap_summary(fg; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(fg))
    # a single `(1 | g)` keeps its bootstrap
    f1 = drm(bf(@formula(yp ~ x + (1 | g))), Poisson(); data = d)
    @test size(bootstrap_summary(f1; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(f1))
end
