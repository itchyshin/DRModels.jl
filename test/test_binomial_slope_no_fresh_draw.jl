using DRModels, Test, Random

# A Binomial fit with a random slope, a slope plus an intercept bar, or crossed bars has no fresh
# random-effect draw. Without a refusal `bootstrap_summary` and `bootstrap_result` fall back to a draw
# with the random effects at zero (the variance-component interval collapses) and return a table, as soon
# as the fitted object stores its trial counts. They refuse by name instead. A single `(1 | g)` and fixed
# effects keep their bootstrap.
function _bsl_data(seed)
    r = Random.Xoshiro(seed); n = 300; G = 20
    gi = rand(r, 1:G, n); x = randn(r, n); z = randn(r, n)
    b0 = 0.4 .* randn(r, G); b1 = 0.3 .* randn(r, G)
    eta = 0.3 .+ 0.4 .* x .+ b0[gi] .+ b1[gi] .* z
    p = 1 ./ (1 .+ exp.(-eta))
    s = Float64[sum(rand(r) < pi for _ in 1:10) for pi in p]
    return (; x, z, g = string.("g", gi), s, f = 10.0 .- s)
end

# Shapes without a fresh draw: one slope term and a slope plus an intercept bar (crossed intercepts have
# a fresh draw, see the last test set). Each must refuse by name through every bootstrap entry point and the default `simulate`,
# whether or not the fitted object stores its trial counts.
_bsl_cases = [
    ("(0 + z | g)", "a random slope without an intercept", "(0 + z | g)", identity),
    ("(1 + z | g)", "a correlated random intercept and slope", "(1 + z | g)", identity),
    ("(1 | g) + (0 + z | g)", "several random-effect terms", "(1 | g) + (0 + z | g)", identity),
]

@testset "Binomial random effects without a fresh draw: bootstrap and default simulate refuse by name" begin
    d0 = _bsl_data(5)
    d = merge(d0, (; h = string.("h", rand(Random.Xoshiro(9), 1:6, length(d0.x)))))
    for (terms, label, shown, _) in _bsl_cases
        aghq = endswith(terms, "AGHQ")
        t = replace(terms, " AGHQ" => "")
        fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula(cbind(s, f) ~ x + $t))"))
        fit = aghq ? drm(fo, Binomial(); data = d, marginal = :AGHQ) : drm(fo, Binomial(); data = d)
        calls = (
            "bootstrap_summary" => () -> bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
            "bootstrap_result" => () -> bootstrap_result(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
            "bootstrap_ci" => () -> bootstrap_ci(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
            "bootstrap_summary(formula)" => () -> bootstrap_summary(fo, Binomial(); data = d, B = 2,
                rng = Random.Xoshiro(1)),
            "bootstrap_result(formula)" => () -> bootstrap_result(fo, Binomial(); data = d, B = 2,
                rng = Random.Xoshiro(1)),
            "default simulate" => () -> simulate(fit; data = d, rng = Random.Xoshiro(1)),
        )
        for (cname, call) in calls
            err = try; call(); nothing; catch e; e; end
            @test err isa ArgumentError
            err isa ArgumentError || continue
            msg = sprint(showerror, err)
            @test occursin("no fresh random-effect draw", msg)
            @test occursin("for Binomial() with $label ($shown)", msg)
            # the text must not depend on whether trials are stored, and must not claim that
            # `re = :zero` fails
            @test !occursin("does not carry trials", msg)
            @test !occursin("does not work", msg)
        end
    end
end

@testset "Binomial refusal leaves the working shapes alone" begin
    d = _bsl_data(5)
    # a single (1 | g) keeps its fresh draw and its bootstrap
    f1 = drm(bf(@formula(cbind(s, f) ~ x + (1 | g))), Binomial(); data = d)
    @test size(bootstrap_summary(f1; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(f1))
    # fixed effects only keep theirs
    f0 = drm(bf(@formula(cbind(s, f) ~ x)), Binomial(); data = d)
    @test size(bootstrap_summary(f0; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(f0))
    # `re = :zero` is not touched by the refusal: it still works for the correlated shape
    f2 = drm(bf(@formula(cbind(s, f) ~ x + (1 + z | g))), Binomial(); data = d)
    @test length(simulate(f2; data = d, re = :zero, rng = Random.Xoshiro(1))) == length(d.x)
end

@testset "Binomial crossed intercepts have a fresh draw: the bootstrap runs and the draws carry the group effects" begin
    d0 = _bsl_data(5)
    d = merge(d0, (; h = string.("h", rand(Random.Xoshiro(9), 1:6, length(d0.x)))))
    gm(y) = [sum(y[d.g .== k]) / count(==(k), d.g) for k in unique(d.g)]
    gv(y) = (m = gm(y); sum(abs2, m .- sum(m) / length(m)) / (length(m) - 1))
    for kw in ((;), (; marginal = :AGHQ))
        fit = drm(bf(@formula(cbind(s, f) ~ x + (1 | g) + (1 | h))), Binomial(); data = d, kw...)
        @test size(bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(fit))
        fresh = sum(gv(simulate(fit; data = d, rng = Random.Xoshiro(s))) for s in 1:40) / 40
        # between-group variance of the simulated group means against the data's (own bound, no R value)
        @test 0.5 < fresh / gv(d.s) < 2
    end
end

@testset "Binomial crossed intercepts: the draws that bootstrap_result itself makes carry the group effects" begin
    using Statistics, StableRNGs
    # No R reference exists for this cell (drmTMB has no bootstrap for this model), so the bounds below come
    # from a simulation, not from a parity number, and are never to be loosened. Fixed StableRNG streams
    # (and `MersenneTwister` seeded from them for the replicates, as the bootstrap seeds them): the same on
    # every Julia. Default marginal only: the AGHQ refits cost about 1.5 s each, and both estimators share
    # the one marginal simulator that this checks.
    d0 = _bsl_data(5)
    d = merge(d0, (; h = string.("h", rand(Random.Xoshiro(9), 1:6, length(d0.x)))))
    gl = unique(d.g)
    gv(y) = var([mean(y[d.g .== k]) for k in gl])
    fit = drm(bf(@formula(cbind(s, f) ~ x + (1 | g) + (1 | h))), Binomial(); data = d)
    res = bootstrap_result(fit; data = d, B = 200, rng = StableRNG(20261114))
    @test res.used >= 60      # refits that do not converge are dropped; about half are, 90 or more are kept
    # Main check 1: the variance of the group means over the bootstrap's own simulated data sets. Replicate b
    # is the marginal simulator run on MersenneTwister(res.seeds[b]). Against the same statistic with the
    # random effects at zero the ratio is 4.6 (20000 data sets simulated independently of the package; 0.1%
    # quantile of the ratio of the two means 4.1); two statistics that both carry no random effects give a
    # ratio of 1.0 (99.9% quantile 1.13). The threshold 2 sits between the two, a factor of 2 from each centre.
    sim = DRModels._marginal_simulator(fit, d)
    vboot = mean(gv(sim(Random.MersenneTwister(sd))) for sd in res.seeds)
    vzero = mean(gv(simulate(fit; data = d, re = :zero, rng = StableRNG(k))) for k in 1:100)
    @test vboot > 2 * vzero
    # Main check 2, on the refits that `bootstrap_summary` reports (`res.summary` is its table): the variance of
    # the refitted intercept against the noise-only value, the squared standard error of the fixed-effect
    # model (no random effect). With the random effects drawn it is 7.7 times that (4000 refits from the fitted
    # parameters; 1% quantile of one B = 200 call 5.4); with them at zero it is 1.1 times (99% quantile 1.6).
    # The threshold 2.5 is a factor of 2.3 above the noise-only centre and 2.2 below the lowest drawn value.
    se_fixed = sqrt(vcov(drm(bf(@formula(cbind(s, f) ~ x)), Binomial(); data = d))[1, 1])
    row(p, c) = only(r for r in res.summary if r.param === p && r.coef == c)
    @test row(:mu, "(Intercept)").std_error^2 > 2.5 * se_fixed^2
    # The lower end of the interval of log SD(g) stays away from the boundary when the effects are drawn (1%
    # quantile -1.56 against a fitted -0.85); draws at zero put it below -15 in 99% of calls.
    @test row(:resd, "g").lower > log(0.1)
end
