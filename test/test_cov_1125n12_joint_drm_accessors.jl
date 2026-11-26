# cov lane 1125n12 — formula joint_drm public imputation and coef aliases.
using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "cov 1125n12: JointDrmFit imputed and coef aliases" begin
    Random.seed!(11251202)
    n = 20
    z = randn(n)
    xtrue = 0.1 .+ 0.55 .* z
    y = 0.15 .+ 0.35 .* z .+ 0.8 .* xtrue .+ 0.2 .* randn(n)
    dat = (; y = Union{Missing,Float64}[i in (3, 11) ? missing : y[i] for i in 1:n],
           x = Union{Missing,Float64}[i in (5, 9, 15) ? missing : xtrue[i] for i in 1:n],
           z)
    form = bf(@formula(y ~ 1 + z + mi(x)), @formula(sigma ~ 1))
    ctl = miss_control(response = "include", predictor = "model")
    fit = drm(form, Gaussian(); data = dat, impute = (x = @formula(x ~ 1 + z),),
              missing = ctl, g_tol = 1e-4)
    @test fit isa JointDrmFit
    @test fit.variable === :x
    @test isfinite(loglik(fit))
    @test length(coef(fit)) == length(coef(fit.prepared.fit))
    @test coef(fit, :mi_x) == coef(fit.prepared.fit, :mi_x)
    @test all(>(0), coef(fit, :sigma_mi_x))
    @test coef(fit, :sigma_mi_x) ≈ exp.(coef(fit, :logsd_mi_x)) rtol = 1e-12
    @test_throws ArgumentError coef(fit, :sigma_mi_xb)
    summary = joint_missing_summary(fit.prepared)
    @test summary.predictor === :gaussian
    @test summary.uncertainty_status === :not_computed
    imp_all = imputed(fit; rows = :all, se = false)
    @test imp_all.variable == fill("x", n)
    @test length(imp_all.estimate) == n
    imp_miss = imputed(fit; rows = :missing)
    @test all(.!imp_miss.observed)
    @test length(imp_miss.estimate) == count(.!fit.prepared.prepared.observed_x)
    @test_throws ArgumentError imputed(fit; variable = :z)
    @test vcov(fit) === vcov(fit.prepared.fit)
    @test nobs(fit) == count(!ismissing, dat.y)
    @test family(fit) isa Gaussian
    @test convergence_status(fit) isa Symbol
end

@testset "cov 1125n12: JointDrmFit Bernoulli predictor aliases" begin
    Random.seed!(11251203)
    n = 18
    z = randn(n)
    xb = [((i % 4) < 2 ? 1.0 : 0.0) for i in 1:n]
    y = -0.1 .+ 0.3 .* z .+ 0.9 .* xb .+ 0.1 .* randn(n)
    dat = (; y = Union{Missing,Float64}[y[i] for i in 1:n],
           xb = Union{Missing,Float64}[i in (4, 12) ? missing : xb[i] for i in 1:n], z)
    form = bf(@formula(y ~ z + mi(xb)), @formula(sigma ~ 1))
    fit = drm(form, Gaussian(); data = dat,
        impute = (xb = impute_model(@formula(xb ~ 1); family = Binomial()),),
        missing = miss_control(predictor = "model"), g_tol = 1e-4)
    @test coef(fit, :mi_xb) == coef(fit, :mi_x)
    @test_throws ArgumentError coef(fit, :sigma_mi_xb)
    @test_throws ArgumentError coef(fit, :logsd_mi_xb)
    table = imputed(fit; variable = "xb", rows = :all)
    @test all(==("xb"), table.variable)
    @test all(ismissing, table.std_error[table.observed])
end

@testset "cov 1125n12: JointTwoDrmFit summary via prepared" begin
    Random.seed!(11251204)
    n = 16
    z = randn(n)
    x1 = 0.2 .+ 0.4 .* z
    x2 = -0.1 .+ 0.3 .* z
    y = 0.1 .+ 0.25 .* z .+ 0.5 .* x1 .+ 0.4 .* x2 .+ 0.15 .* randn(n)
    dat = (; y = Union{Missing,Float64}[y[i] for i in 1:n],
           x1 = Union{Missing,Float64}[i == 6 ? missing : x1[i] for i in 1:n],
           x2 = Union{Missing,Float64}[i == 10 ? missing : x2[i] for i in 1:n], z)
    form = bf(@formula(y ~ z + mi(x1) + mi(x2)), @formula(sigma ~ 1))
    ctl = miss_control(response = "include", predictor = "model")
    fit = drm(form, Gaussian(); data = dat,
        impute = (x1 = @formula(x1 ~ z), x2 = @formula(x2 ~ 1)), missing = ctl, g_tol = 1e-4)
    @test fit isa JointTwoDrmFit
    @test fit.variables == (:x1, :x2)
    summary = joint_missing_summary(fit.prepared)
    @test summary.predictor_variables == (:x1, :x2)
    @test isfinite(loglik(fit))
    @test coef(fit, :sigma_mi_x1) ≈ exp.(coef(fit, :logsd_mi_x1)) rtol = 1e-12
    imp = imputed(fit; variable = :x2, rows = :missing, se = false)
    @test all(==("x2"), imp.variable)
    @test !any(imp.observed)
end