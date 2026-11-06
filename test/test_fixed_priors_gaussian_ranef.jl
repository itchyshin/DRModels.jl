using Test, DRModels, ForwardDiff, LinearAlgebra

@testset "fixed priors admit a Gaussian mean random intercept" begin
    dat = (y=[1., 1.4, 2.0, 2.7, 1.2, 1.9, 3.2, 3.4],
           x=[0., 1., 0., 1., 0., 1., 0., 1.],
           id=repeat(1:4, inner=2))
    form = bf(@formula(y ~ x + (1 | id)))
    priors = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=0.2, sd=2))
    rows = resolve_priors(form, Gaussian(); data=dat, priors=priors)
    @test [(r.dpar, r.coef) for r in rows] == [(:mu, "x")]
    fit = drm(form, Gaussian(); data=dat, priors=priors)
    @test fit.estim_method == :MAP
    @test isfinite(fit.nll(fit.theta))
    plain = drm(form, Gaussian(); data=dat)
    point = [0.7, 0.4, -0.2, -0.5]
    prior_value(t) = 0.5*((t[2]-0.2)/2)^2 + log(2) + 0.5log(2π)
    @test fit.nll(point) ≈ plain.nll(point) + prior_value(point) atol=1e-10
    @test ForwardDiff.gradient(fit.nll, point) ≈ ForwardDiff.gradient(plain.nll, point) + ForwardDiff.gradient(prior_value, point) atol=1e-9
    @test ForwardDiff.hessian(fit.nll, point) ≈ ForwardDiff.hessian(plain.nll, point) + ForwardDiff.hessian(prior_value, point) atol=1e-9
    @test fit.loglik ≈ -plain.nll(fit.theta) atol=1e-10
    @test_throws ArgumentError confint(fit; method=:wald)
    @test !isempty(profile_targets(fit))
    @test all(r -> !r.profile_ready, profile_targets(fit))
    @test isempty(profile_targets(fit; ready_only=true))
    fixed_only = drm(bf(@formula(y ~ x)), Gaussian(); data=dat, priors=priors)
    @test !isempty(profile_targets(fixed_only))
    @test all(r -> !r.profile_ready, profile_targets(fixed_only))
    @test_throws ArgumentError drm(form, Gaussian(); data=dat, priors=priors, marginal=:Laplace)
    @test_throws ArgumentError drm(form, Gaussian(); data=merge(dat, (;y=[NaN; dat.y[2:end]])), priors=priors)
    missing_group = merge(dat, (;id=Union{Missing,Int}[missing; dat.id[2:end]]))
    missing_x = merge(dat, (;x=Union{Missing,Float64}[missing; dat.x[2:end]]))
    @test_throws ArgumentError resolve_priors(form, Gaussian(); data=missing_group, priors=priors)
    @test_throws ArgumentError drm(form, Gaussian(); data=missing_group, priors=priors)
    @test_throws ArgumentError drm(form, Gaussian(); data=missing_x, priors=priors)
    @test_throws ArgumentError drm(form, Gaussian(); data=dat, priors=priors, method=:REML)
    @test_throws ArgumentError drm(form, Gaussian(); data=dat, priors=priors, marginal=:VA)
    @test_throws ArgumentError drm(form, Gaussian(); data=dat, priors=priors, algorithm=:lbfgs)
    @test_throws ArgumentError drm(form, Gaussian(); data=dat, priors=priors, K=Matrix{Float64}(I, 4, 4))
    for c in (DRMControl(maxit=1), DRMControl(start=[0., 0., 0., 0.]), DRMControl(multi_start=2))
        @test_throws ArgumentError drm(form, Gaussian(); data=dat, priors=priors, control=c)
    end
    controlled = drm(form, Gaussian(); data=dat, priors=priors,
                     control=DRMControl(optimizer=:bfgs, se=false))
    info = control_info(controlled)
    @test info.actual_optimizer === :bfgs
    @test all(isnan, controlled.vcov)
    selected = only(filter(run -> run.stage === info.selected_optimizer_stage, info.optimizer_runs))
    @test selected.objective ≈ controlled.nll(controlled.theta) atol=1e-10
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | id))), Gaussian(); data=dat, priors=priors)
    opts = Dict("priors"=>[Dict("kind"=>"normal", "dpar"=>"mu", "coef"=>"x", "mean"=>0.2, "sd"=>2.)],
                "prior_expected"=>[Dict("dpar"=>"mu", "coef"=>"x")])
    bridge = drm_bridge(; formula="y ~ x + (1 | id)", family="gaussian", data=dat, options=opts)
    @test bridge["penalized_objective"] ≈ fit.nll(fit.theta) atol=1e-8
    @test_throws ArgumentError drm_bridge(; formula="y ~ x + (1 | id)", family="gaussian", data=dat,
        options=merge(opts, Dict("prior_expected"=>[Dict("dpar"=>"mu", "coef"=>"wrong")])))
    zero = drm_priors(drm_prior_ridge(dpar="mu", coef="x", lambda=0))
    zerofit = drm(form, Gaussian(); data=dat, priors=zero)
    emptyfit = drm(form, Gaussian(); data=dat, priors=drm_priors())
    @test zerofit.estim_method == :ML
    @test !isempty(profile_targets(zerofit; ready_only=true))
    @test zerofit.theta ≈ plain.theta atol=1e-9
    @test emptyfit.theta ≈ plain.theta atol=1e-9
    @test zerofit.nll(point) ≈ plain.nll(point) atol=1e-12
    @test zerofit.loglik ≈ plain.loglik atol=1e-9
    @test plain.penalty === nothing
    @test haskey(ranef(fit), :id)
    @test haskey(vc(fit), :id)
    refitted = update(fit, form; data=dat)
    @test refitted.theta ≈ fit.theta atol=1e-8
    @test refitted.nll(point) ≈ fit.nll(point) atol=1e-10
    both = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=0.2, sd=2),
                      drm_prior_ridge(dpar="sigma", coef="x", lambda=0.5))
    locscale = bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ x))
    sf = drm(locscale, Gaussian(); data=dat, priors=both)
    su = drm(locscale, Gaussian(); data=dat)
    q = [0.7, 0.4, -0.2, 0.1, -0.5]
    @test sf.nll(q) ≈ su.nll(q) + prior_value(q) + 0.25*q[4]^2 atol=1e-10
end
