# cov lane 1125n11 — fast public drm() REML integration (Gaussian + Poisson, small n).
using DRModels
using Test, Random, Statistics, LinearAlgebra

_msg(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

@testset "cov 1125n11: Gaussian location–scale REML accessors" begin
    Random.seed!(1125114)
    n = 48
    x = randn(n)
    y = 1.0 .+ 0.6 .* x .+ 0.9 .* randn(n)
    data = (; y, x)
    form = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    fml = drm(form, Gaussian(); data, method = :ML)
    freml = drm(form, Gaussian(); data, method = :REML)
    @test estimation_method(fml) === :ML
    @test estimation_method(freml) === :REML
    @test isnan(reml_loglik(fml))
    @test isfinite(reml_loglik(freml))
    @test loglik(freml) == reml_loglik(freml)
    @test isfinite(ml_loglik(freml))
    @test ml_loglik(freml) <= loglik(fml) + 1e-7
    @test isfinite(deviance(freml))
    @test dof_residual(freml) > 0
    @test dof_residual(freml) ≤ n
    @test isfinite(aic(freml))
    @test isfinite(bic(freml))
    @test all(isfinite, fitted(freml))
    @test all(isfinite, predict(freml, data))
    @test length(residuals(freml)) == n
    @test exp(2 * coef(freml, :sigma)[1]) > exp(2 * coef(fml, :sigma)[1])
end

@testset "cov 1125n11: Gaussian ordinary (1 | g) REML" begin
    Random.seed!(1125115)
    G, m = 8, 3
    g = repeat(1:G, inner = m)
    n = G * m
    x = randn(n)
    b = 0.5 .* randn(G)
    y = 0.2 .+ 0.4 .* x .+ b[g] .+ 0.7 .* randn(n)
    data = (; y, x, g)
    form = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
    fml = drm(form, Gaussian(); data, method = :ML, g_tol = 1e-3)
    freml = drm(form, Gaussian(); data, method = :REML, g_tol = 1e-3)
    @test estimation_method(freml) === :REML
    @test nobs(freml) == n
    @test isfinite(reml_loglik(freml))
    @test isfinite(ml_loglik(freml))
    @test haskey(re_sd(freml), :g)
    @test re_sd(freml)[:g] ≥ re_sd(fml)[:g] - 1e-6
    @test length(ranef(freml)[:g]) == G
    V = vcov(freml)
    @test size(V) == (length(coef(freml)), length(coef(freml)))
    @test all(isfinite, V)
    @test convergence_status(freml) isa Symbol
end

@testset "cov 1125n11: Poisson (1 | g) opt-in REML" begin
    Random.seed!(1125116)
    G, m = 7, 4
    g = repeat(1:G, inner = m)
    n = G * m
    x = randn(n)
    b = 0.4 .* randn(G)
    y = [max(0, round(Int, exp(0.2 + 0.15 * x[i] + b[g[i]] + 0.1 * randn()))) for i in 1:n]
    data = (; y = Float64.(y), x, g)
    form = bf(@formula(y ~ x + (1 | g)))
    fml = drm(form, Poisson(); data, g_tol = 1e-3)
    freml = drm(form, Poisson(); data, method = :REML, g_tol = 1e-3)
    @test estimation_method(fml) === :ML
    @test estimation_method(freml) === :REML
    @test isfinite(freml.reml_loglik)
    @test isfinite(freml.ml_loglik)
    @test freml.loglik == freml.reml_loglik
    @test freml.reml_loglik != freml.ml_loglik
    @test all(isfinite, fitted(freml))
    @test all(fitted(freml) .> 0)
    sb_ml = exp(coef(fml)[end])
    sb_reml = exp(coef(freml)[end])
    @test sb_reml ≥ sb_ml - 1e-6
end

@testset "cov 1125n11: REML refusals on unsupported drm routes" begin
    Random.seed!(1125117)
    n = 30
    x = randn(n)
    g = repeat(1:5, inner = 6)
    y = abs.(x .+ randn(n))
    d = (; y, x, g)
    msg_nb = _msg(() -> drm(bf(@formula(y ~ x + (1 | g))), NegBinomial2(); data = d, method = :REML))
    @test occursin("REML", msg_nb)
    msg_bb = _msg(() -> drm(bf(@formula(y ~ x + (1 | g))), BetaBinomial(); data = d, method = :REML))
    @test occursin("REML", msg_bb)
    msg_va = _msg(() -> drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = d, method = :VA))
    @test occursin("marginal", msg_va) || occursin("Laplace", msg_va)
end

@testset "cov 1125n11: REML model comparison guard (same mean)" begin
    Random.seed!(1125118)
    n = 40
    x = randn(n)
    y = 0.5 .+ x .+ randn(n)
    data = (; y, x)
    full = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data, method = :REML)
    redσ = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data, method = :REML)
    @test estimation_method(full) === :REML
    @test estimation_method(redσ) === :REML
    cmp = lrtest(full, redσ)
    @test hasproperty(cmp, :statistic)
    @test isfinite(cmp.statistic)
    @test isfinite(cmp.pvalue)
    diff_mean = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data, method = :REML)
    msg = _msg(() -> lrtest(full, diff_mean))
    @test occursin("REML", msg) || occursin("mean", lowercase(msg))
end