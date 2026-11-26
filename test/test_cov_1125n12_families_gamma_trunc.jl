# cov lane 1125n12 — Gamma / Beta / TruncatedPoisson small-n drm routes.
using DRModels
using Test, Random, Statistics

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

@testset "cov 1125n12: Gamma fixed effects" begin
    Random.seed!(11251207)
    n = 36
    x = randn(n)
    μ = exp.(0.3 .+ 0.4 .* x)
    y = [max(0.05, μ[i] * (1 + 0.35 * randn())) for i in 1:n]
    dat = (; y, x)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gamma(); data = dat, g_tol = 1e-4)
    @test fit.family isa Gamma
    @test all(>(0), fitted(fit))
    @test all(>(0), predict(fit, dat))
    @test isfinite(loglik(fit))
    @test isfinite(aic(fit))
    shape = exp(-2 * only(coef(fit, :sigma)))
    @test shape > 0 && isfinite(shape)
    @test convergence_status(fit) isa Symbol
    @test nobs(fit) == n
    msg = _msg(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma(); data = (y = [0.0, 1.0], x = [0.0, 1.0])))
    @test occursin("positive", lowercase(msg))
    msg2 = _msg(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma(); data = dat, method = :REML))
    @test occursin("REML", msg2) || occursin("method", lowercase(msg2))
end

@testset "cov 1125n12: Beta fixed effects" begin
    Random.seed!(11251208)
    n = 32
    x = randn(n)
    p = 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x)))
    y = [clamp(p[i] + 0.08 * randn(), 0.02, 0.98) for i in 1:n]
    dat = (; y, x)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Beta(); data = dat, g_tol = 1e-4)
    @test fit.family isa Beta
    @test all(>(0), fitted(fit))
    @test all(<(1), fitted(fit))
    @test isfinite(loglik(fit))
    φ = exp(-2 * only(coef(fit, :sigma)))
    @test φ > 0 && isfinite(φ)
    bad = _msg(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Beta();
        data = (y = [0.0, 0.5, 1.0], x = [0.0, 0.0, 0.0])))
    @test occursin("interval", lowercase(bad)) || occursin("(0,1)", bad) || occursin("0,1", bad)
end

@testset "cov 1125n12: TruncatedPoisson fixed effects" begin
    Random.seed!(11251209)
    n = 28
    x = randn(n)
    λ = exp.(0.15 .+ 0.25 .* x)
    y = Float64[max(1, round(Int, λ[i] * (0.85 + 0.1 * rand()))) for i in 1:n]
    dat = (; y, x)
    @test all(y .>= 1)
    fit = drm(bf(@formula(y ~ 1 + x)), TruncatedPoisson(); data = dat, g_tol = 1e-4)
    @test fit.family isa TruncatedPoisson
    @test all(fitted(fit) .> 0)
    @test isfinite(loglik(fit))
    @test length(residuals(fit)) == n
    msg = _msg(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x)), TruncatedPoisson(); data = dat))
    @test occursin("sigma", lowercase(msg)) || occursin("fixed effects", lowercase(msg))
    msg2 = _msg(() -> drm(bf(@formula(y ~ x)), TruncatedPoisson(); data = dat, nAGQ = 3))
    @test occursin("nAGQ", msg2) || occursin("SD", msg2) || occursin("random", lowercase(msg2))
end