using Test, DRModels
using StatsModels: @formula

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

@testset "Unsupported random structures identify SD priors" begin
    x = repeat([-1.0, 0.0, 1.0], 4)
    g = repeat(1:4; inner=3)
    d = (y=repeat([0.0, 1.0], 6), x=x, g=g, h=copy(g))
    sdprior = drm_prior_sd_gamma(dpar="mu", group="g", shape=2, rate=3)
    fixed = drm_prior_normal(dpar="mu", coef="x", sd=2)
    formulas = (bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ x)),
                bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | h))))
    for f in formulas, priors in (drm_priors(sdprior), drm_priors(fixed, sdprior))
        err = try
            drm(f, Gaussian(); data=d, priors=priors)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("SD priors", sprint(showerror, err))
        @test !occursin("fixed-effect priors", sprint(showerror, err))
    end
    sd_model_error = try
        drm(formulas[1], Gaussian(); data=d, priors=drm_priors(sdprior))
        nothing
    catch e
        e
    end
    @test sd_model_error isa ArgumentError
    @test occursin("SD submodels", sprint(showerror, sd_model_error))
    err = try
        drm(formulas[2], Gaussian(); data=d, priors=drm_priors(fixed))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("fixed-effect priors currently do not support random or structured effects", sprint(showerror, err))

    # Additional unsupported-route diagnostics retain fixed-only compatibility.
    x = repeat([-1.0, 0.0, 1.0], 4)
    g = repeat(1:4; inner=3)
    d = (y=repeat([0.0, 1.0], 6), x=x, g=g)
    f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
    sdprior = drm_prior_sd_gamma(dpar="mu", group="g", shape=2, rate=3)
    fixed = drm_prior_normal(dpar="mu", coef="x", sd=2)
    for priors in (drm_priors(sdprior), drm_priors(fixed, sdprior))
        err = try
            drm(f, Gaussian(); data=d, priors=priors, method=:REML)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("SD priors", sprint(showerror, err))
        @test occursin("ML", sprint(showerror, err))
        @test !occursin("fixed-effect priors", sprint(showerror, err))
    end
    err = try
        drm(f, Gaussian(); data=d, priors=drm_priors(fixed), method=:REML)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("fixed-effect priors currently require ordinary fixed-effect Gaussian ML", sprint(showerror, err))
    nan_data = merge(d, (y=[NaN; d.y[2:end]],))
    rank_data = merge(d, (z=copy(d.x),))
    rank_formula = bf(@formula(y ~ x + z + (1 | g)), @formula(sigma ~ 1))
    for priors in (drm_priors(sdprior), drm_priors(fixed, sdprior))
        weight_error = try
            drm(f, Gaussian(); data=d, priors=priors, weights=[2.0; ones(11)])
            nothing
        catch e
            e
        end
        @test weight_error isa ArgumentError
        @test occursin("weights", sprint(showerror, weight_error))
        for (model, data, kwargs, expected) in (
            (f, nan_data, (;), "complete responses"),
            (rank_formula, rank_data, (;), "full-rank"),
        )
            err = try
                drm(model, Gaussian(); data=data, priors=priors, kwargs...)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("SD priors", sprint(showerror, err))
            @test occursin(expected, sprint(showerror, err))
        end
    end

end
