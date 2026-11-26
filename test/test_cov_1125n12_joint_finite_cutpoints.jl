# cov lane 1125n12 — JointFiniteDrmFit cutpoints and wrapper accessors.
using DRModels
using Test, Random, LinearAlgebra

@testset "cov 1125n12: ordinal cutpoints and joint_missing_summary wrapper" begin
    Random.seed!(11251205)
    levels = ["low", "mid", "high"]
    n = 24
    z = randn(n)
    xraw = [levels[1 + mod(i, 3)] for i in 1:n]
    y = 0.1 .+ 0.3 .* z .+ 0.2 .* randn(n)
    dat = (; y = Union{Missing,Float64}[i == 7 ? missing : y[i] for i in 1:n],
           x = Union{Missing,String}[i in (5, 14, 20) ? missing : xraw[i] for i in 1:n], z)
    form = bf(@formula(y ~ 1 + z + mi(x)), @formula(sigma ~ 1))
    ctl = miss_control(response = "include", predictor = "model")
    fit = drm(form, Gaussian(); data = dat,
        impute = (x = impute_model(@formula(x ~ z); family = CumulativeLogit(), levels = levels),),
        missing = ctl, g_tol = 1e-4)
    @test fit isa JointFiniteDrmFit
    raw = coef(fit, :rawcut_x)
    cuts = cutpoints(fit)
    @test length(cuts) == length(raw)
    @test cuts[1] ≈ raw[1] rtol = 1e-14
    @test cuts[2] ≈ cuts[1] + exp(raw[2]) rtol = 1e-12
    @test all(cuts[2:end] .> cuts[1:end - 1])
    summary_wrap = joint_missing_summary(fit)
    summary_prep = joint_missing_summary(fit.prepared)
    @test summary_wrap.conditional_probabilities ≈ summary_prep.conditional_probabilities
    @test isfinite(loglik(fit))
    @test fitted(fit; re = :conditional) == fitted(fit.prepared.fit; re = :conditional)
    imp = imputed(fit; variable = "x", rows = :all)
    @test imp.variable == fill("x", n)
    @test length(imp.estimate) == n
end

@testset "cov 1125n12: categorical wrapper refuses cutpoints" begin
    Random.seed!(11251206)
    levels = ["a", "b", "c"]
    n = 15
    z = randn(n)
    dat = (; y = Union{Missing,Float64}[0.2 + 0.1 * randn() for _ in 1:n],
           x = Union{Missing,String}[i == 4 ? missing : levels[1 + mod(i, 3)] for i in 1:n], z)
    form = bf(@formula(y ~ z + mi(x)), @formula(sigma ~ 1))
    fit = drm(form, Gaussian(); data = dat,
        impute = (x = impute_model(@formula(x ~ 1); family = CategoricalLogit(), levels = levels),),
        missing = miss_control(predictor = "model"), g_tol = 1e-4)
    @test_throws ArgumentError coef(fit, :rawcut_x)
    @test_throws ArgumentError cutpoints(fit)
    @test coef(fit, :mi_x) == coef(fit.prepared.fit, :mi_x)
end

@testset "cov 1125n12: impute_model levels contract" begin
    imp_ord = impute_model(@formula(x ~ z); family = CumulativeLogit(), levels = ["1", "2", "3"])
    @test imp_ord.levels == ["1", "2", "3"]
    @test_throws ArgumentError impute_model(@formula(x ~ z); family = Gaussian(), levels = ["a"])
    @test_throws ArgumentError impute_model(@formula(x ~ z); family = CumulativeLogit(), levels = ["dup", "dup", "x"])
    @test_throws ArgumentError impute_model(@formula(x ~ z); family = CumulativeLogit(), levels = ["only", "two"])
end