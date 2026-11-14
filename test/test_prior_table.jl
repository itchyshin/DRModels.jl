using Test, DRModels, ForwardDiff

# Tables.jl column tables are accepted directly via the existing `priors` keyword.
# Expected cases are transcribed from the independently reviewed table contract;
# they do not depend on any R implementation.
const pt_data = (y=[.2,.8,.5,1.4,1.0,1.6,2.0,1.3,2.2,1.8,2.4,2.1],
                 x=[-1.,0.,1.,-1.,0.,1.,-1.,0.,1.,-1.,0.,1.],
                 site=repeat(["a","b","c","d"], inner=3),
                 observer=repeat(["u","v","w"], 4),
                 year=repeat(["i","j"], 6))
const pt_form = bf(@formula(y ~ x + (1 | site) + (1 | observer)))
const pt_reverse = bf(@formula(y ~ x + (1 | observer) + (1 | site)))
const pt_fixed = bf(@formula(y ~ x))
pt(prior, class, coef; kwargs...) = (; prior=prior, class=class, coef=coef, kwargs...)
preview(form, fam, data, p) = resolve_priors(form, fam; data=data, priors=p)

@testset "prior table public contract" begin
    @testset "fixed selection and legacy constructors" begin
        table = pt(["normal(0,2)", "normal(1,3)"], ["fixef", "fixef"], ["", "(Intercept)"])
        legacy = drm_priors(drm_prior_normal(dpar="mu", coef="all", intercept="exclude", mean=0, sd=2),
                            drm_prior_normal(dpar="mu", coef="(Intercept)", mean=1, sd=3))
        got = preview(pt_fixed, Gaussian(), pt_data, table)
        expected = preview(pt_fixed, Gaussian(), pt_data, legacy)
        @test [(r.dpar,r.coef,r.precision,r.constant) for r in got] ==
              [(r.dpar,r.coef,r.precision,r.constant) for r in expected]
        fit = drm(pt_fixed, Gaussian(); data=pt_data, priors=table)
        ref = drm(pt_fixed, Gaussian(); data=pt_data, priors=legacy)
        q = [0.4, 0.7, -0.3]
        @test fit.nll(q) ≈ ref.nll(q) atol=1e-10
        @test fit.theta ≈ ref.theta atol=1e-6
        @test ForwardDiff.gradient(fit.nll,q) ≈ ForwardDiff.gradient(ref.nll,q) atol=1e-8
        @test ForwardDiff.hessian(fit.nll,q) ≈ ForwardDiff.hessian(ref.nll,q) atol=1e-8
        plain = drm(pt_fixed, Gaussian(); data=pt_data)
        @test plain.nll(q) ≈ drm(pt_fixed, Gaussian(); data=pt_data, priors=nothing).nll(q) atol=1e-12
        zero = pt(["ridge(0)"], ["fixef"], ["x"])
        zfit = drm(pt_fixed, Gaussian(); data=pt_data, priors=zero)
        zref = drm(pt_fixed, Gaussian(); data=pt_data, priors=drm_priors(drm_prior_ridge(dpar="mu",coef="x",lambda=0)))
        @test zfit.nll(q) ≈ zref.nll(q) atol=1e-12
        @test zfit.nll(q) ≈ plain.nll(q) atol=1e-12
        @test zfit.estim_method == zref.estim_method
        @test isequal(confint(zfit), confint(zref))
        @test aic(zfit) ≈ aic(zref)
        @test_throws ArgumentError drm(pt_fixed, Gaussian(); data=pt_data, priors=zero, weights=fill(2., length(pt_data.y)))
    end
    @testset "named and broad SD targets" begin
        broad = pt(["gamma(0.6666666666666666,2)"], ["ranef_sd"], [""])
        legacy = drm_priors(drm_prior_sd_gamma(dpar="mu",group="site",shape=2,rate=3),
                            drm_prior_sd_gamma(dpar="mu",group="observer",shape=2,rate=3))
        for form in (pt_form, pt_reverse)
            rows = preview(form, Gaussian(), pt_data, broad)
            refrows = preview(form, Gaussian(), pt_data, legacy)
            @test Dict(r.group => (r.index,r.shape,r.rate) for r in rows) == Dict(r.group => (r.index,r.shape,r.rate) for r in refrows)
            @test Set(r.group for r in rows) == Set(["site", "observer"])
            @test all(r -> r.scale == :sd && r.coef == "(Intercept)", rows)
        end
        specific = pt(["gamma(1,3)", "gamma(0.6666666666666666,2)"],
                      ["ranef_sd", "ranef_sd"], ["observer", "site"])
        expected = drm_priors(drm_prior_sd_gamma(dpar="mu",group="observer",shape=3,rate=3),
                              drm_prior_sd_gamma(dpar="mu",group="site",shape=2,rate=3))
        got = preview(pt_form, Gaussian(), pt_data, specific)
        ref = preview(pt_form, Gaussian(), pt_data, expected)
        @test [(r.group,r.shape,r.rate,r.index) for r in got] == [(r.group,r.shape,r.rate,r.index) for r in ref]
        explicit = pt(["gamma(1,3)"], ["ranef_sd"], ["(Intercept)"]; group=["observer"])
        @test only(preview(pt_form, Gaussian(), pt_data, explicit)).group == "observer"
        mixed = pt(["gamma(1,3)", "normal(0,2)", "gamma(0.6666666666666666,2)"],
                   ["ranef_sd", "fixef", "ranef_sd"], ["observer", "x", "site"])
        mref = drm_priors(drm_prior_sd_gamma(dpar="mu",group="observer",shape=3,rate=3),
                          drm_prior_normal(dpar="mu",coef="x",mean=0,sd=2),
                          drm_prior_sd_gamma(dpar="mu",group="site",shape=2,rate=3))
        fit = drm(pt_form, Gaussian(); data=pt_data, priors=mixed)
        reference = drm(pt_form, Gaussian(); data=pt_data, priors=mref)
        q = [.4,.7,-.3,log(.8),log(.4)]
        @test fit.nll(q) ≈ reference.nll(q) atol=1e-10
        @test ForwardDiff.gradient(fit.nll,q) ≈ ForwardDiff.gradient(reference.nll,q) atol=1e-8
        @test ForwardDiff.hessian(fit.nll,q) ≈ ForwardDiff.hessian(reference.nll,q) atol=1e-8
        @test fit.nll(fit.theta) ≈ reference.nll(reference.theta) atol=1e-6
        @test update(fit, pt_form; data=pt_data).nll(q) ≈ fit.nll(q) atol=1e-9
        @test_throws ArgumentError confint(fit)
        @test_throws ArgumentError aic(fit)
    end
    @testset "fixed-only binomial remains supported" begin
        d = (y=[0,0,0,1,0,1,1,1], x=[-3.,-2.,-1.,0.,.5,1.,2.,3.])
        table = pt(["ridge(1)","ridge(1)"], ["fixef","fixef"], ["(Intercept)","x"])
        old = drm_priors(drm_prior_ridge(dpar="mu",coef="all",intercept="include",lambda=1))
        a = drm(pt_fixed, Binomial(); data=d, priors=table)
        b = drm(pt_fixed, Binomial(); data=d, priors=old)
        @test a.theta ≈ b.theta atol=1e-6
        @test a.nll(a.theta) ≈ b.nll(b.theta) atol=1e-8
    end
    @testset "invalid tables and selectors fail before fitting" begin
        invalid = Any[pt([s],["fixef"],["x"]) for s in
                   ("normal(0,2);system('x')","normal(1+1,2)","normal(0,Inf)","normal(0,NaN)","normal(0,1e-999)","normal(0,1e999)","normal(0)","normal(0,0)","normal(0,-1)","normal(0,1e200)","normal(0,1e-200)","normal(0,1e155)","normal(0,1e-155)","ridge(-1)","ridge(1,2)","gamma(1,2) trailing")]
        append!(invalid, [pt(["gamma(0,2)"],["ranef_sd"],["site"]),
            pt(["gamma(1,1)"],["ranef_sd"],["site"]),
            pt(["gamma(1e-320,2)"],["ranef_sd"],["site"]),
            pt(["normal(0,2)"],["ranef_cor"],["x"]),
            pt(["normal(0,2)"],["fixef_disp"],["x"]),
            pt(["normal(0,2)"],["fixef_zi"],["x"]),
            pt(["normal(0,2)"],["unknown"],["x"]),
            pt(["normal(0,2)"],["fixef"],["x"];dpar=["unknown"]),
            pt(["normal(0,2)","normal(0,2)"],["fixef","fixef"],["x","x"]),
            pt(["gamma(1,2)","gamma(1,2)"],["ranef_sd","ranef_sd"],["site","site"]),
            pt(["gamma(1,2)"],["fixef"],["x"]),
            pt(["normal(0,2)"],["fixef"],["all"]),
            pt(["normal(0,2)","normal(0,2)"],["fixef","fixef"],["","x"]),
            pt(["gamma(1,2)","gamma(1,2)"],["ranef_sd","ranef_sd"],["","site"]),
            pt(["gamma(1,2)"],["ranef_sd"],["site"];group=["observer"]),
            pt(["gamma(1,2)"],["ranef_sd"],["x"];group=["site"]),
            pt(["gamma(1,2)"],["ranef_sd"],["site"];dpar=["sigma"]),
            pt(["gamma(1,2)"],["ranef_sd"],["unknown"]),
            pt(["normal(0,2)"],["fixef"],["x"];group=["site"]),
            pt(["normal(0,2)"],["fixef"],[missing]),
            (prior=["normal(0,2)"],class=["fixef"]),
            (prior=["normal(0,2)"],class=["fixef"],coef=["x"],unexpected=["value"]),
            pt(String[],String[],String[])])
        for bad in invalid
            @test_throws ArgumentError preview(pt_form, Gaussian(), pt_data, bad)
        end
        @test_throws ArgumentError preview(bf(@formula(y ~ 1)), Gaussian(), pt_data, pt(["ridge(1)"],["fixef"],[""]))
        @test_throws ArgumentError preview(bf(@formula(y ~ all)), Gaussian(), merge(pt_data,(all=pt_data.x,)), pt(["ridge(1)"],["fixef"],["all"]))
    end
    @testset "real model eligibility" begin
        prior = pt(["gamma(1,2)"],["ranef_sd"],[""])
        @test_throws ArgumentError preview(pt_fixed, Gaussian(), pt_data, prior)
        @test_throws ArgumentError preview(bf(@formula(y ~ x + (1+x | site))), Gaussian(), pt_data, prior)
        @test_throws ArgumentError preview(bf(@formula(y ~ x + (1 | site) + (1 | observer) + (1 | year))), Gaussian(), pt_data, prior)
        @test_throws ArgumentError preview(pt_form, Gaussian(), merge(pt_data,(observer=pt_data.site,)), prior)
        @test_throws ArgumentError preview(pt_form, Binomial(), pt_data, prior)
        @test_throws ArgumentError drm(pt_form, Gaussian(); data=pt_data, priors=prior, weights=fill(2.,length(pt_data.y)))
        missingdata = merge(pt_data,(y=Union{Missing,Float64}[missing; pt_data.y[2:end]],))
        @test_throws ArgumentError drm(pt_form, Gaussian(); data=missingdata, priors=prior)
    end
end
println("JULIA_PRIOR_TABLE_TESTS_PASS")
