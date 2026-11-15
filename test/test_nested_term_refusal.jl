# `(1 | g / h)` and the other compound groupings after a random-effect bar (`g & h`, `h %in% g`,
# `g:h`, `f(g)`) used to leak an internal error: "type FunctionTerm has no field sym" (an
# ErrorException on Julia 1.10, a FieldError on 1.12 and later) from `_split_ranef`, in every family.
# drmTMB refuses the same terms ("Random-effect grouping terms must be simple variables"), and so
# does drm() now, with a message that names the term and the spelling that works.
# Supporting `g / h` itself is a new feature and is NOT done here (owner decision pending).
using DRModels, Test, Random

grp_wrap(x) = x     # a user function after the bar: `(1 | grp_wrap(g))`

function _nested_data(n = 48)
    Random.seed!(11)
    g = repeat(string.("g", 1:6), 8); h = repeat(string.("h", 1:2), inner = 24)
    return (; y = randn(n), yp = Float64.(rand(0:5, n)), yb = Float64.(rand(n) .< 0.5), yg = rand(n) .+ 0.5,
            y2 = randn(n), x1 = randn(n), g, h, gh = string.(g, ":", h))
end

function _refusal(f, fam; kwargs...)
    d = _nested_data()
    try
        drm(f, fam; data = d, kwargs...)
        return nothing
    catch e
        return e
    end
end

@testset "compound random-effect groupings are refused by name" begin
    cases = [
        ("g / h",        bf(@formula(y ~ x1 + (1 | g / h))),                "g / h",     "(1 | g) + (1 | g_h)"),
        ("g & h",        bf(@formula(y ~ x1 + (1 | g & h))),                "g & h",     "(1 | g_h)"),
        ("h %in% g",     bf(@formula(y ~ x1 + (1 | g) + (1 | h %in% g))),   "h %in% g",  "(1 | g_h)"),
        ("g:h",          bf(@formula(y ~ x1 + (1 | g:h))),                  "g:h",       "(1 | g_h)"),
        ("slope g / h",  bf(@formula(y ~ x1 + (0 + x1 | g / h))),           "g / h",     "(0 + x1 | g) + (0 + x1 | g_h)"),
        ("sigma g / h",  bf(@formula(y ~ x1), @formula(sigma ~ 1 + (1 | g / h))), "g / h", "(1 | g) + (1 | g_h)"),
    ]
    for (label, f, shown, spelled) in cases
        e = _refusal(f, Gaussian())
        @test e isa ArgumentError
        e isa ArgumentError || (println("  case `$label` gave: ", typeof(e)); continue)
        @test occursin(shown, e.msg)
        @test occursin("not supported", e.msg)
        @test occursin("Random-effect grouping terms must be simple variables", e.msg)
        @test occursin(spelled, e.msg)
        @test !occursin("FunctionTerm", e.msg)
    end
    # a function call after the bar has no spelled alternative, but is still a named refusal
    e = _refusal(bf(@formula(y ~ x1 + (1 | grp_wrap(g)))), Gaussian())
    @test e isa ArgumentError
    e isa ArgumentError && @test occursin("grp_wrap(g)", e.msg) && occursin("plain column", e.msg)
end

@testset "every univariate family gives the same refusal" begin
    d = _nested_data()
    for (fam, f) in ((Gaussian(), bf(@formula(y ~ x1 + (1 | g / h)))),
                     (Poisson(), bf(@formula(yp ~ x1 + (1 | g / h)))),
                     (Binomial(), bf(@formula(yb ~ x1 + (1 | g / h)))),
                     (NegBinomial2(), bf(@formula(yp ~ x1 + (1 | g / h)))),
                     (Gamma(), bf(@formula(yg ~ x1 + (1 | g / h)))),
                     (Beta(), bf(@formula(yg ~ x1 + (1 | g / h)))),
                     (LogNormal(), bf(@formula(yg ~ x1 + (1 | g / h)))),
                     (Student(), bf(@formula(y ~ x1 + (1 | g / h)))))
        e = try drm(f, fam; data = d); nothing catch err; err end
        @test e isa ArgumentError
        e isa ArgumentError && @test occursin("g / h", e.msg)
    end
end

@testset "bivariate and mixed-family routes refuse it too" begin
    f = bf(mu1 = @formula(y ~ x1 + (1 | g / h)), mu2 = @formula(y2 ~ x1))
    for fam in (Gaussian(), (Gaussian(), Gamma()))
        d = merge(_nested_data(), (; y2 = abs.(_nested_data().y2) .+ 0.5))
        e = try drm(f, fam; data = d); nothing catch err; err end
        @test e isa ArgumentError
        e isa ArgumentError && @test occursin("g / h", e.msg)
    end
end

@testset "the suggested spelling fits" begin
    d = _nested_data()
    for (fam, f) in ((Gaussian(), bf(@formula(y ~ x1 + (1 | g) + (1 | gh)))),
                     (Poisson(), bf(@formula(yp ~ x1 + (1 | g) + (1 | gh)))),
                     (LogNormal(), bf(@formula(yg ~ x1 + (1 | g) + (1 | gh)))))
        fit = drm(f, fam; data = d)
        @test fit isa DRModels.DrmFit
        @test isfinite(loglikelihood(fit))
    end
end

@testset "a plain grouping and a labelled block are not refused" begin
    d = _nested_data()
    fit = drm(bf(@formula(y ~ x1 + (1 | g))), Gaussian(); data = d)
    @test fit isa DRModels.DrmFit
    fit2 = drm(bf(@formula(y ~ x1 + (1 + x1 | g))), Gaussian(); data = d)
    @test fit2 isa DRModels.DrmFit
end

@testset "a structured marker as the group of a labelled bar is not refused" begin
    # `(1 | p | phylo(species))` is a documented spelling; the refusal must not catch it.
    f = bf(@formula(y ~ x1 + (1 | p | phylo(g))), @formula(sigma ~ 1 + (1 | p | phylo(g))))
    d = _nested_data()
    e = try drm(f, Gaussian(); data = d); nothing catch err; err end
    @test !(e isa ArgumentError && occursin("not supported (drmTMB refuses it too", e.msg))
end
