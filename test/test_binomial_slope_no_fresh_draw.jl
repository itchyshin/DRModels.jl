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
