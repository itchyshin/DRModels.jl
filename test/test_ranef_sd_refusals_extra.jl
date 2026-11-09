using Test, DRModels, StatsModels

@testset "Gamma SD prior rejects invalid selectors and parameters" begin
    for a in (1.0, 0.0, -1.0, NaN, Inf), b in (3.0,)
        @test_throws ArgumentError drm_prior_sd_gamma(dpar="mu", group="g", shape=a, rate=b)
    end
    for b in (0.0, -1.0, NaN, Inf)
        @test_throws ArgumentError drm_prior_sd_gamma(dpar="mu", group="g", shape=2, rate=b)
    end
    @test_throws ArgumentError drm_prior_sd_gamma(dpar="sigma", group="g", shape=2, rate=3)
    @test_throws ArgumentError drm_prior_sd_gamma(dpar="mu", group="", shape=2, rate=3)
    @test_throws ArgumentError drm_prior_sd_gamma(dpar="mu", group="g", coef="all", shape=2, rate=3)
end

@testset "Gamma SD prior refuses unsupported target and model shapes" begin
    x = repeat([-1.0, 0.0, 1.0], 4)
    g = repeat(1:4; inner=3)
    d = (y=repeat([0.0, 1.0], 6), x=x, g=g, h=copy(g))
    p = drm_priors(drm_prior_sd_gamma(dpar="mu", group="g", shape=2, rate=3))
    f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
    @test length(resolve_priors(f, Gaussian(); data=d, priors=p)) == 1
    # The group variable, not merely its observed values, identifies the target.
    wrong = drm_priors(drm_prior_sd_gamma(dpar="mu", group="h", shape=2, rate=3))
    @test_throws ArgumentError resolve_priors(f, Gaussian(); data=d, priors=wrong)
    @test_throws ArgumentError drm(f, Gaussian(); data=d, priors=wrong)
    @test_throws ArgumentError drm(f, Binomial(); data=d, priors=p)
    @test_throws ArgumentError drm(f, Gaussian(); data=d, priors=p, method=:REML)
    @test_throws ArgumentError drm(f, Gaussian(); data=d, priors=p, weights=[2.0; ones(11)])
    @test_throws ArgumentError resolve_priors(bf(@formula(y ~ x + (1 + x | g))), Gaussian(); data=d, priors=p)
    @test_throws ArgumentError resolve_priors(bf(@formula(y ~ x + (1 | g) + (1 | h))), Gaussian(); data=d, priors=p)
    @test_throws ArgumentError resolve_priors(bf(@formula(y ~ x + phylo(1 | g))), Gaussian(); data=d, priors=p)
    @test_throws ArgumentError resolve_priors(bf(@formula(y ~ x + relmat(1 | g))), Gaussian(); data=d, priors=p)
    # Labelled terms are rejected by the formula grammar before target resolution.
    labelled_error = try
        resolve_priors(bf(@formula(y ~ x + (1 | block | g))), Gaussian(); data=d, priors=p)
        nothing
    catch err
        err
    end
    @test labelled_error isa ErrorException
    @test occursin("unsupported random-effect term", sprint(showerror, labelled_error))
    for column in (:y, :x, :g)
        original = getproperty(d, column)
        values = Union{Missing,eltype(original)}[missing; original[2:end]]
        incomplete = merge(d, NamedTuple{(column,)}((values,)))
        @test_throws ArgumentError resolve_priors(f, Gaussian(); data=incomplete, priors=p)
    end
end
