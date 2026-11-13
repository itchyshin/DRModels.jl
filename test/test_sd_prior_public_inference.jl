using Test, DRModels

@testset "Public inference refuses an SD-prior MAP fit" begin
    data = (y=[0.5897018631581462,2.0205043733267987,0.9312667881951244,1.112187614910214,2.3629251775024493,1.1484035222055766,1.9232040823661078,2.271753470357653,0.636777735776596],
            x=[-1.2,0.2,-0.8,0.1,1.1,-1.0,-0.3,0.6,1.3], g=[1,1,2,2,2,3,3,3,3])
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x))
    prior = drm_priors(drm_prior_sd_gamma(dpar="mu", group="g", shape=2., rate=3.))
    fit = drm(formula, Gaussian(); data=data, priors=prior)
    plain = drm(formula, Gaussian(); data=data)
    @test fit.estim_method === :MAP
    for call in (() -> bootstrap_result(fit; data=data, B=2),
                 () -> confint(fit; method=:profile), () -> profile_result(fit),
                 () -> predict(fit, data; se=true), () -> predict_parameters(fit, data; se=true),
                 () -> aic(fit), () -> bic(fit), () -> lrt_boundary(fit, plain; q=1),
                 () -> DRModels.lrtest(plain, fit))
        failure = try call(); nothing catch e; e end
        @test failure isa ArgumentError
        @test occursin(r"prior|MAP"i, sprint(showerror, failure))
    end
    err = try DRModels.lrtest(plain, fit); nothing catch e; e end
    @test occursin("priors or penalties", sprint(showerror, err))
    table = coeftable(fit)
    @test all(col -> all(isnan, col), table.cols[3:6])
end
