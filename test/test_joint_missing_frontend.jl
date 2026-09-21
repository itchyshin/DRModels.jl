using Test
using DRModels
using LinearAlgebra

@testset "joint missing-predictor formula frontend" begin
    BLAS.set_num_threads(1)
    @test BLAS.get_num_threads() == 1
    @test isdefined(DRModels, :mi)
    @test isdefined(DRModels, :miss_control)
    @test isdefined(DRModels, :impute_model)
    @test isdefined(DRModels, :JointDrmFit)

    ctl = DRModels.miss_control(response = "include", predictor = "model")
    @test ctl.response === :include
    @test ctl.predictor === :model
    @test_throws ArgumentError DRModels.miss_control(response = "drop")
    @test_throws ArgumentError DRModels.miss_control(predictor = "impute")
    @test_throws ArgumentError mi(1)

    n = 32
    z = collect(range(-1.2, 1.2; length = n))
    xfull = 0.15 .+ 0.65 .* z .+ 0.15 .* sin.(1:n)
    yfull = 0.3 .+ 0.4 .* z .+ 0.7 .* xfull .+ 0.18 .* cos.(1:n)
    dat = (; y = Union{Missing,Float64}[i in (4, 24) ? missing : yfull[i] for i in 1:n],
             x = Union{Missing,Float64}[i in (8, 18, 24) ? missing : xfull[i] for i in 1:n],
             z,
             grp = repeat(["a", "b", "c", "d"], 8))
    form = DRModels.bf(@formula(y ~ z + mi(x)), @formula(sigma ~ 1))
    imp = (x = @formula(x ~ z),)

    @test DRModels._has_joint_mi(form)
    @test !DRModels._has_joint_mi(DRModels.bf(@formula(y ~ z), @formula(sigma ~ 1)))

    # Estimated local run: well under two minutes, with one BLAS thread.
    fit = drm(form, Gaussian(); data = dat, impute = imp, missing = ctl, g_tol = 1e-8)
    @test fit isa DRModels.JointDrmFit
    @test fit.variable === :x
    @test length(DRModels.coef(fit)) == 7
    @test length(DRModels.coef(fit, :mu)) == 3
    @test length(DRModels.coef(fit, :mi_x)) == 2
    @test length(DRModels.coef(fit, :sigma_mi_x)) == 1
    @test DRModels.coef(fit, :sigma_mi_x)[1] > 0
    @test size(DRModels.vcov(fit)) == (7, 7)
    @test isposdef(Symmetric(DRModels.vcov(fit)))
    @test DRModels.nobs(fit) == 30
    @test isfinite(DRModels.loglik(fit))
    @test DRModels.family(fit) isa DRModels.Gaussian
    @test DRModels.is_converged(fit)
    @test DRModels.niterations(fit) >= 0

    @test_throws ArgumentError DRModels._fit_joint_formula(
        DRModels.bf(@formula(y ~ z * mi(x)), @formula(sigma ~ 1)), dat;
        impute = imp, missing = ctl)
    @test_throws ArgumentError DRModels._fit_joint_formula(
        DRModels.bf(@formula(y ~ mi(x)), @formula(sigma ~ mi(x))), dat;
        impute = imp, missing = ctl)
    @test_throws ArgumentError DRModels._fit_joint_formula(
        DRModels.bf(@formula(y ~ z + mi(x) + mi(w)), @formula(sigma ~ 1)), merge(dat, (; w = dat.x));
        impute = imp, missing = ctl)
    @test_throws ArgumentError DRModels._fit_joint_formula(
        DRModels.bf(@formula(y ~ z + mi(x) + (1 | grp)), @formula(sigma ~ 1)), dat;
        impute = imp, missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = dat,
        impute = (x = DRModels.impute_model(@formula(x ~ z); family = DRModels.Binomial()),),
        missing = ctl, method = :REML)
    @test_throws ArgumentError drm(form, Gaussian(); data = dat,
        impute = imp, missing = DRModels.miss_control(response = "include", predictor = "fail"))
    @test_throws ArgumentError drm(form, Gaussian(); data = dat,
        impute = imp, missing = ctl, algorithm = :em)
    @test_throws ArgumentError DRModels._fit_joint_formula(
        DRModels.bf(@formula(y ~ 0 + mi(x)), @formula(sigma ~ 1)), dat;
        impute = imp, missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = dat,
        impute = (x = @formula(z ~ z),), missing = ctl)
    @test_throws ArgumentError drm(form, Gaussian(); data = dat,
        impute = (x = @formula(x ~ z), extra = @formula(x ~ z)), missing = ctl)
    incomplete = merge(dat, (; z = Union{Missing,Float64}[i == 3 ? missing : z[i] for i in 1:n]))
    @test_throws ArgumentError DRModels._fit_joint_formula(form, incomplete; impute = imp, missing = ctl)

    # A Bernoulli predictor is a separate exact likelihood route, not a
    # Gaussian predictor coerced to 0/1.
    xbfull = Float64[((i % 4 == 1 || i % 5 == 0) ? 1 : 0) for i in 1:n]
    ybfull = -0.1 .+ 0.25 .* z .+ 0.85 .* xbfull .+ 0.15 .* cos.(1:n)
    bdat = (; y = Union{Missing,Float64}[ybfull[i] for i in 1:n],
              xb = Union{Missing,Float64}[i in (7, 15, 24) ? missing : xbfull[i] for i in 1:n], z)
    bform = DRModels.bf(@formula(y ~ z + mi(xb)), @formula(sigma ~ 1))
    bfit = drm(bform, Gaussian(); data = bdat,
        impute = (xb = DRModels.impute_model(@formula(xb ~ z); family = DRModels.Binomial()),),
        missing = DRModels.miss_control(predictor = "model"), g_tol = 1e-8)
    @test DRModels.is_converged(bfit)
    @test isposdef(Symmetric(DRModels.vcov(bfit)))
    @test DRModels.coef(bfit, :mi_xb) == DRModels.coef(bfit.prepared.fit, :mi_x)
    @test_throws ArgumentError DRModels.coef(bfit, :sigma_mi_xb)

    factordat = merge(dat, (; f = repeat(["low", "mid", "high", "mid"], 8)))
    factorform = DRModels.bf(@formula(y ~ f + mi(x)), @formula(sigma ~ 1))
    _, factor_rhs = DRModels._joint_mean_parts(Dict(factorform.forms)[:mu])
    _, Xfactor, factor_names = DRModels._design(:y, factor_rhs, factordat)
    @test size(Xfactor, 2) == 3
    @test length(factor_names) == 3

    richform = DRModels.bf(@formula(y ~ log(z + 2) + z & f + mi(x)), @formula(sigma ~ 1))
    _, rich_rhs = DRModels._joint_mean_parts(Dict(richform.forms)[:mu])
    _, Xrich, _ = DRModels._design(:y, rich_rhs, factordat)
    @test size(Xrich, 2) > 3
end

@testset "joint public routing refuses ignored arguments" begin
    d=(y=[1.0,2.0,3.0],x=[0.0,1.0,2.0])
    plain=bf(@formula(y ~ x))
    @test_throws ArgumentError drm(plain,Gaussian();data=d,impute=(x=@formula(x ~ 1),))
    @test_throws ArgumentError drm(plain,Gaussian();data=d,missing=miss_control())
    @test_throws ArgumentError drm(bf(@formula(y ~ x),@formula(sigma ~ mi(x))),Gaussian();data=d,impute=(x=@formula(x ~ 1),),missing=miss_control(predictor="model"))
end
println("JOINT_FRONTEND_PUBLIC_PASS")

@testset "joint model requires exogenous complete covariates" begin
    n=32;z=collect(range(-1.2,1.2;length=n))
    x=0.15 .+ 0.65 .* z .+ 0.15 .* sin.(1:n)
    y=0.3 .+ 0.4 .* z .+ 0.7 .* x .+ 0.18 .* cos.(1:n)
    dat=(y=y,x=x,z=z)
    form=bf(@formula(y ~ z + mi(x)))
    ctl=miss_control(predictor="model")
    @test_throws ArgumentError drm(form,Gaussian();data=dat,impute=(x=@formula(x ~ z),),missing=JointMissingControl(:drop,:model))
    for invalid in (bf(@formula(y ~ x + mi(x))),
                    bf(@formula(y ~ y + mi(x))),
                    bf(@formula(y ~ mi(x)),@formula(sigma ~ x)),
                    bf(@formula(y ~ mi(x)),@formula(sigma ~ y)))
        @test_throws ArgumentError drm(invalid,Gaussian();data=dat,impute=(x=@formula(x ~ z),),missing=ctl)
    end
    for invalid in (@formula(x ~ x),@formula(x ~ y),@formula(x ~ log(abs(y)) + z))
        @test_throws ArgumentError drm(form,Gaussian();data=dat,impute=(x=invalid,),missing=ctl)
    end
    @test_throws ArgumentError drm(bf(@formula(y ~ mi(y))),Gaussian();data=dat,impute=(y=@formula(y ~ z),),missing=ctl)
end
println("JOINT_EXOGENOUS_REFUSALS_PASS")
