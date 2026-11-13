# A bare `@formula` (no `bf()`) and R's `x1:x2` used to end in raw MethodErrors:
#   drm(@formula(y ~ x), Gaussian(); data)  -> "no method matching drm(::FormulaTerm, ::Gaussian; ...)"
#   y ~ x1 + x2 + x1:x2                     -> "Cannot convert ... StepRangeLen ... to Float64"
# Both are named ArgumentErrors now. Spellings that fail before `drm` runs (`poly()`, `I()`,
# `splines::ns()`) cannot be caught by the package and are not tested here.
using DRModels, Test, Random

function _spelling_data(n = 48)
    Random.seed!(5)
    g = repeat(string.("g", 1:6), 8)
    return (; y = randn(n), yp = Float64.(rand(0:5, n)), yg = rand(n) .+ 0.5, y2 = randn(n) .+ 3,
            x1 = randn(n), x2 = randn(n), g)
end

_err(thunk) = try thunk(); nothing catch e; e end

@testset "a bare formula without bf() is a named refusal" begin
    d = _spelling_data()
    f = @formula(y ~ x1 + x2)
    for fam in (Gaussian(), Student(), Poisson(), Binomial(), NegBinomial2(), Gamma(), Beta(), LogNormal(),
                Tweedie(), SkewNormal(), CumulativeLogit(), (Gaussian(), Gamma()))
        e = _err(() -> drm(f, fam; data = d))
        @test e isa ArgumentError
        e isa ArgumentError || (println("  family ", fam, " gave: ", typeof(e)); continue)
        @test occursin("must be created with `bf()`", e.msg)
        @test occursin("bf(@formula(y ~ x1 + x2))", e.msg)
        @test occursin("drmTMB", e.msg)
    end
    # the same message with keywords, with no data, and with a keyword no route takes
    for call in (() -> drm(f, Gaussian()),
                 () -> drm(f, Gaussian(); data = d, method = :REML),
                 () -> drm(f, Gaussian(); data = d, control = DRMControl(se = false)),
                 () -> drm(f, Gaussian(); data = d, nonsense = 1))
        e = _err(call)
        @test e isa ArgumentError && occursin("must be created with `bf()`", e.msg)
    end
    e = _err(() -> drm(f, Poisson(); data = d))
    @test e isa ArgumentError && occursin("Poisson()", e.msg)
    e = _err(() -> drm(f, (Gaussian(), Gamma()); data = d))
    @test e isa ArgumentError && occursin("(Gaussian(), Gamma())", e.msg)
end

@testset "R's x1:x2 is a named refusal" begin
    d = _spelling_data()
    cases = [
        ("mu, after a sum",   bf(@formula(y ~ x1 + x2 + x1:x2)),                   "x1 + x2 + x1:x2"),
        ("mu, alone",         bf(@formula(y ~ x1:x2)),                             "x1:x2"),
        ("mu, middle",        bf(@formula(y ~ x1 + x1:x2 + x2)),                   "x1 + x1:x2 + x2"),
        ("sigma",             bf(@formula(y ~ x1), @formula(sigma ~ x1 + x2 + x1:x2)), "x1 + x2 + x1:x2"),
        ("with a random term", bf(@formula(y ~ x1 + x2 + x1:x2 + (1 | g))),         "x1:x2"),
    ]
    for (label, f, shown) in cases
        e = _err(() -> drm(f, Gaussian(); data = d))
        @test e isa ArgumentError
        e isa ArgumentError || (println("  case `$label` gave: ", typeof(e)); continue)
        @test occursin(shown, e.msg)
        @test occursin("range", e.msg)
        @test occursin("a & b", e.msg) && occursin("a * b", e.msg)
        @test !occursin("StepRange", e.msg)
    end
    for (fam, f) in ((Poisson(),      bf(@formula(yp ~ x1 + x2 + x1:x2))),
                     (Gamma(),        bf(@formula(yg ~ x1 + x2 + x1:x2))),
                     (Student(),      bf(@formula(y ~ x1 + x2 + x1:x2))),
                     (LogNormal(),    bf(@formula(yg ~ x1 + x2 + x1:x2))),
                     (Binomial(),     bf(@formula(yp ~ x1 + x2 + x1:x2))))
        e = _err(() -> drm(f, fam; data = d))
        @test e isa ArgumentError && occursin("x1:x2", e.msg)
    end
    fb = bf(mu1 = @formula(y ~ x1 + x2 + x1:x2), mu2 = @formula(y2 ~ x1))
    for fam in (Gaussian(), (Gaussian(), Gamma()))
        d2 = merge(d, (; y2 = abs.(d.y2) .+ 0.5))
        e = _err(() -> drm(fb, fam; data = d2))
        @test e isa ArgumentError && occursin("x1:x2", e.msg)
    end
end

@testset "the nested-grouping refusal for g:h keeps its own message" begin
    e = _err(() -> drm(bf(@formula(y ~ x1 + (1 | g:x2))), Gaussian(); data = _spelling_data()))
    @test e isa ArgumentError && occursin("Random-effect grouping terms must be simple variables", e.msg)
end

@testset "the supported spellings still fit" begin
    d = _spelling_data()
    for f in (bf(@formula(y ~ x1 + x2 + x1 & x2)), bf(@formula(y ~ x1 * x2)), bf(@formula(y ~ x1 + x2)))
        fit = drm(f, Gaussian(); data = d)
        @test fit isa DRModels.DrmFit && isfinite(loglikelihood(fit))
    end
end
