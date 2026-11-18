using Test, DRModels, LinearAlgebra, ForwardDiff, Logging

@testset "fixed-effect priors: selection and density" begin
    p = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=1, sd=2),
                   drm_prior_ridge(dpar="sigma", coef="all", intercept="exclude", lambda=0.5))
    names = [:mu => ["(Intercept)", "x"], :sigma => ["(Intercept)", "x"]]
    rows = DRModels._resolve_fixed_priors(p, names, Dict(:mu => 1, :sigma => 1))
    @test length(rows) == 2
    @test [(r.dpar, r.coef, r.index) for r in rows] == [(:mu, "x", 2), (:sigma, "x", 4)]
    theta = [9.0, 3.0, 8.0, 2.0]
    oracle = 0.5 + log(2) + 0.5log(2π) + 0.5 * 0.5 * 2^2
    @test DRModels._fixed_prior_value(rows, theta) ≈ oracle atol=1e-12
    @test ForwardDiff.gradient(t -> DRModels._fixed_prior_value(rows, t), theta) ≈ [0, .5, 0, 1]
    @test diag(ForwardDiff.hessian(t -> DRModels._fixed_prior_value(rows, t), theta)) ≈ [0, .25, 0, .5]
    @test_throws ArgumentError drm_prior_normal(dpar="mu", coef="all", sd=1)
    @test_throws ArgumentError drm_prior_ridge(dpar="mu", coef="x", lambda=-1)
    @test_throws ArgumentError DRModels._resolve_fixed_priors(drm_priors(
        drm_prior_ridge(dpar="mu", coef="all", intercept="include", lambda=1),
        drm_prior_normal(dpar="mu", coef="x", sd=1)), names, Dict(:mu => 1, :sigma => 1))
end

@testset "real design column resolution" begin
    dat = (y=[1., 2., 2.2, 4., 4.1, 6.2, 6.8, 8.], x=collect(0.:7.),
           f=["a", "b", "a", "b", "a", "b", "a", "b"])
    form = bf(@formula(y ~ x + f + x & f))
    p = drm_priors(drm_prior_ridge(dpar="mu", coef="all", intercept="exclude", lambda=.25))
    cache_before = copy(form.schema_cache[])
    rows = resolve_priors(form, Gaussian(); data=dat, priors=p)
    @test form.schema_cache[] == cache_before
    @test length(rows) == 3
    @test all(r -> r.coef != "(Intercept)", rows)
    fit = drm(form, Gaussian(); data=dat, priors=p)
    @test [(r.dpar, r.coef) for r in fit.penalty.rows] == [(r.dpar, r.coef) for r in rows]
    @test any(r -> occursin("&", r.coef) || occursin(":", r.coef), rows)
    noint = bf(@formula(y ~ 0 + f))
    factorrows = resolve_priors(noint, Gaussian(); data=dat, priors=p)
    @test length(factorrows) == 2
    @test all(r -> r.coef != "(Intercept)", factorrows)
    factorfit = drm(noint, Gaussian(); data=dat, priors=p)
    @test [r.coef for r in factorfit.penalty.rows] == [r.coef for r in factorrows]
    alias = (y=dat.y, x=dat.x, z=copy(dat.x))
    bad = bf(@formula(y ~ x + z))
    @test_throws ArgumentError resolve_priors(bad, Gaussian(); data=alias, priors=p)
    @test_throws ArgumentError drm(bad, Gaussian(); data=alias, priors=p)
end

@testset "two dpars and binomial bridge agreement" begin
    dat = (y=[1., 2.2, 2.9, 4.1, 4.8, 6.3, 6.7, 8.1], x=collect(0.:7.))
    p = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=.2, sd=2),
                   drm_prior_normal(dpar="sigma", coef="x", mean=-.1, sd=1.5))
    f = bf(@formula(y ~ x), @formula(sigma ~ x))
    fit = drm(f, Gaussian(); data=dat, priors=p)
    q = [.5, .9, -.1, .08]; X = hcat(ones(length(dat.y)), dat.x)
    manual(t) = sum(X*t[3:4] .+ .5 .* (dat.y .- X*t[1:2]).^2 .* exp.(-2 .* (X*t[3:4]))) + length(dat.y)*.5log(2π) + .5*((t[2]-.2)/2)^2 + log(2)+.5log(2π) + .5*((t[4]+.1)/1.5)^2 + log(1.5)+.5log(2π)
    @test fit.nll(q) ≈ manual(q) atol=1e-10
    h=1e-5; eye=Matrix{Float64}(I,4,4)
    fd = [(manual(q+h*eye[:,j])-manual(q-h*eye[:,j]))/(2h) for j in 1:4]
    @test maximum(abs.(ForwardDiff.gradient(fit.nll,q)-fd)) < 1e-6
    fdh = hcat([(ForwardDiff.gradient(manual,q+h*eye[:,j])-ForwardDiff.gradient(manual,q-h*eye[:,j]))/(2h) for j in 1:4]...)
    @test maximum(abs.(ForwardDiff.hessian(fit.nll,q)-fdh)) < 1e-5
    opts = Dict("priors"=>[Dict("kind"=>"normal", "dpar"=>"mu", "coef"=>"x", "mean"=>.2, "sd"=>2.), Dict("kind"=>"normal", "dpar"=>"sigma", "coef"=>"x", "mean"=>-.1, "sd"=>1.5)],
                "prior_expected"=>[Dict("dpar"=>"mu", "coef"=>"x"), Dict("dpar"=>"sigma", "coef"=>"x")])
    out = drm_bridge(; formula="y ~ x; sigma ~ x", family="gaussian", data=dat, options=opts)
    @test out["coefficients"] ≈ fit.theta atol=1e-7
    @test out["penalized_objective"] ≈ fit.nll(fit.theta) atol=1e-10

    bd = (y=[0, 0, 0, 1, 1, 1], x=[-3., -2., -1., 1., 2., 3.])
    bp = drm_priors(drm_prior_ridge(dpar="mu", coef="all", intercept="include", lambda=1))
    bfitted = drm(bf(@formula(y ~ x)), Binomial(); data=bd, priors=bp)
    bo = drm_bridge(; formula="y ~ x", family="binomial", data=bd, options=Dict(
        "priors"=>[Dict("kind"=>"ridge", "dpar"=>"mu", "coef"=>"all", "intercept"=>"include", "lambda"=>1.)],
        "prior_expected"=>[Dict("dpar"=>"mu", "coef"=>"(Intercept)"), Dict("dpar"=>"mu", "coef"=>"x")]))
    @test bo["coefficients"] ≈ bfitted.theta atol=1e-7
    @test bo["penalized_objective"] ≈ bfitted.nll(bfitted.theta) atol=1e-10
    partial = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=0, sd=2))
    partialfit = @test_logs (:warn, r"leaves some coefficients unpenalized") drm(bf(@formula(y ~ x)), Binomial(); data=bd, priors=partial)
    partialbridge = @test_logs (:warn, r"leaves some coefficients unpenalized") drm_bridge(; formula="y ~ x", family="binomial", data=bd, options=Dict(
        "priors"=>[Dict("kind"=>"normal", "dpar"=>"mu", "coef"=>"x", "mean"=>0., "sd"=>2.)],
        "prior_expected"=>[Dict("dpar"=>"mu", "coef"=>"x")]))
    @test partialbridge["coefficients"] ≈ partialfit.theta atol=1e-7
    @test partialbridge["penalized_objective"] ≈ partialfit.nll(partialfit.theta) atol=1e-10
end

@testset "independent ridge oracle and matched bridge" begin
    dat = (y=[1., 2.1, 2.7, 4.3, 4.8, 6.2, 6.8, 8.3], x=collect(0.:7.))
    p = drm_priors(drm_prior_ridge(dpar="mu", coef="all", intercept="include", lambda=0.4))
    fit = drm(bf(@formula(y ~ x)), Gaussian(); data=dat, priors=p)
    X = hcat(ones(length(dat.y)), dat.x)
    variance = exp(2fit.theta[3])
    exact = (X'X / variance + 0.4I) \ (X'dat.y / variance)
    @test fit.theta[1:2] ≈ exact atol=1e-7
    q = [0.4, 1.1, -0.1]
    fref = sum(q[3] .+ 0.5 .* (dat.y .- X*q[1:2]).^2 .* exp(-2q[3])) + length(dat.y)*0.5log(2π) + .2sum(abs2,q[1:2])
    @test fit.nll(q) ≈ fref atol=1e-10
    h = 1e-5
    eye = Matrix{Float64}(I, length(q), length(q))
    fd = [(fit.nll(q + h*eye[:,j]) - fit.nll(q - h*eye[:,j]))/(2h) for j in eachindex(q)]
    @test maximum(abs.(fd - ForwardDiff.gradient(fit.nll, q))) < 1e-6
    opts = Dict("priors"=>[Dict("kind"=>"ridge", "dpar"=>"mu", "coef"=>"all", "intercept"=>"include", "lambda"=>.4)],
                "prior_expected"=>[Dict("dpar"=>"mu", "coef"=>"(Intercept)"), Dict("dpar"=>"mu", "coef"=>"x")])
    b = drm_bridge(; formula="y ~ x", family="gaussian", data=dat, options=opts)
    @test b["fixed_prior_contract"] == "fixed_effect_priors_v1"
    @test b["coefficients"] ≈ fit.theta atol=1e-7
    @test b["penalized_objective"] ≈ fit.nll(fit.theta) atol=1e-10
    @test b["likelihood_gradient"] + b["prior_gradient"] ≈ b["penalized_gradient"] atol=1e-10
    z = deepcopy(opts); z["priors"][1]["lambda"] = 0.
    zfit = drm_bridge(; formula="y ~ x", family="gaussian", data=dat, options=z)
    @test zfit["estim_method"] == "ML"
    @test zfit["fixed_prior_contract"] == "fixed_effect_priors_v1"
end

@testset "bridge checks resolved prior coordinates" begin
    dat = (y=[1., 2.2, 2.9, 4.1, 4.8, 6.3], x=collect(0.:5.))
    prior = Dict("kind"=>"normal", "dpar"=>"mu", "coef"=>"x", "mean"=>0., "sd"=>2.)
    expected = [Dict("dpar"=>"mu", "coef"=>"x")]
    opts = Dict("priors"=>[prior], "prior_expected"=>expected)
    out = drm_bridge(; formula="y ~ x", family="gaussian", data=dat, options=opts)
    @test out["estim_method"] == "MAP"
    @test length(out["resolved_priors"]) == 1
    @test isfinite(out["fixed_prior_penalty"])
    @test length(out["penalized_gradient"]) == length(out["coefficients"])
    @test_throws ArgumentError drm_bridge(; formula="y ~ x", family="gaussian", data=dat,
        options=Dict("priors"=>[prior], "prior_expected"=>[Dict("dpar"=>"mu", "coef"=>"wrong")]))
end

@testset "binomial logit prior and route refusals" begin
    dat = (y = [0, 0, 0, 1, 1, 1], x = [-3., -2., -1., 1., 2., 3.])
    p = drm_priors(drm_prior_ridge(dpar="mu", coef="all", intercept="include", lambda=1))
    fit = drm(bf(@formula(y ~ x)), Binomial(); data=dat, priors=p)
    @test fit.estim_method == :MAP
    @test isnan(fit.phylo_penalty)
    b = fit.theta
    eta = b[1] .+ b[2] .* dat.x
    nll = sum(log1p.(exp.(eta)) .- dat.y .* eta)
    @test fit.nll(b) ≈ nll + sum(abs2, b)/2 atol=1e-8
    @test maximum(abs.(ForwardDiff.gradient(fit.nll, b))) < 1e-4
    @test_throws ArgumentError drm(bf(@formula(y ~ x)), Binomial(:probit); data=dat, priors=p)
    @test_throws ArgumentError drm(bf(@formula(y ~ x)), Binomial(); data=dat, priors=p, weights=fill(2.,length(dat.y)))
    zero = drm_priors(drm_prior_ridge(dpar="mu", coef="x", lambda=0))
    @test_throws ArgumentError drm(bf(@formula(y ~ x)), Binomial(); data=dat, priors=zero, weights=fill(2.,length(dat.y)))
    plain, zerofit = with_logger(NullLogger()) do
        (drm(bf(@formula(y ~ x)), Binomial(); data=dat),
         drm(bf(@formula(y ~ x)), Binomial(); data=dat, priors=zero))
    end
    @test zerofit.estim_method == :ML
    @test zerofit.geometry == plain.geometry
    @test zerofit.theta ≈ plain.theta atol=1e-7
    @test isequal(zerofit.vcov, plain.vcov)
    @test keys(zerofit.formula.schema_cache[]) == keys(plain.formula.schema_cache[])
    @test predict(zerofit, dat) ≈ predict(plain, dat)
end

@testset "Gaussian prior enters the fitted objective" begin
    dat = (y = [1.0, 2.2, 2.9, 4.1, 4.8, 6.3], x = collect(0.0:5.0))
    p = drm_priors(drm_prior_normal(dpar="mu", coef="x", mean=0, sd=2),
                   drm_prior_normal(dpar="sigma", coef="(Intercept)", mean=0, sd=3))
    preview = resolve_priors(bf(@formula(y ~ x)), Gaussian(); data=dat, priors=p)
    @test [(r.dpar, r.coef) for r in preview] == [(:mu, "x"), (:sigma, "(Intercept)")]
    fit = drm(bf(@formula(y ~ x)), Gaussian(); data=dat, priors=p)
    @test fit.estim_method == :MAP
    b = fit.theta
    residual = dat.y .- (b[1] .+ b[2] .* dat.x)
    data_nll = sum(b[3] .+ 0.5 .* residual.^2 .* exp(-2b[3])) + length(dat.y)*0.5log(2π)
    prior_nll = .5*(b[2]/2)^2 + log(2) + .5log(2π) + .5*(b[3]/3)^2 + log(3) + .5log(2π)
    @test fit.loglik ≈ -data_nll atol=1e-8
    @test fit.nll(b) ≈ data_nll + prior_nll atol=1e-8
    @test maximum(abs.(ForwardDiff.gradient(fit.nll, b))) < 1e-4
    @test_throws ArgumentError aic(fit)
    @test_throws ArgumentError drm(bf(@formula(y ~ x)), Gaussian(); data=dat, priors=p, weights=fill(2.,length(dat.y)))
    zerop = drm_priors(drm_prior_ridge(dpar="mu", coef="x", lambda=0))
    @test_throws ArgumentError drm(bf(@formula(y ~ x)), Gaussian(); data=dat, priors=zerop, weights=fill(2.,length(dat.y)))
    @test_throws ArgumentError profile_result(fit)
    @test_throws ArgumentError profile_curve(fit, 1)
    @test_throws ArgumentError parameter_surface(fit, 1, 2)
    @test_throws ArgumentError bias_correct(fit, first)
    @test_throws ArgumentError confint(fit; method=:wald)
    @test_throws ArgumentError confint(fit; method=:profile)
    ct = coeftable(fit)
    @test all(isfinite, ct.cols[1])
    @test all(col -> all(isnan, col), ct.cols[3:6])
    @test all(col -> all(isnan, col), summary(fit).cols[3:6])
    @test occursin("MAP fit", sprint(show, MIME("text/plain"), fit))
    @test_throws ArgumentError simulate(fit)
    @test_throws ArgumentError predict(fit, dat; se=true)
    refit = update(fit, fit.formula; data=dat)
    @test refit.estim_method == :MAP
    @test refit.theta ≈ fit.theta atol=1e-6
    ordinary = update(fit, fit.formula; data=dat, method=:ML)
    @test ordinary.estim_method == :ML
    replacement = drm_priors(drm_prior_ridge(dpar="mu", coef="x", lambda=1))
    updated = update(fit, fit.formula; data=dat, method=:ML, priors=replacement)
    @test updated.estim_method == :MAP
    @test length(updated.penalty.rows) == 1
    @test updated.penalty.rows[1].precision == 1
end
