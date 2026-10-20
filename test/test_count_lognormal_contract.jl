# Contract suite for fixed-effect LogNormal(), Poisson() and NegBinomial2() fits
# through drm(): input refusals with their messages, fitted / predict_parameters on
# both scales, residuals, vcov / stderror / confint, aic / bic arithmetic,
# simulate, and loglik(fit) against an independent Distributions.jl sum of logpdf
# at predict_parameters. Test-only: no src/ change. Random effects, structured
# markers, marginal=, offset() and the Poisson / NB2 boundary cases are covered
# elsewhere and deliberately left out.
using DRModels
using Test, Random, Statistics, LinearAlgebra
import Distributions

# Shared accessor contract. `ll_ref(pp, y)` is the independent log-likelihood.
function _contract_checks(fit, data, y, k, ll_ref; support)
    n = length(y)
    @test is_converged(fit)
    @test length(coef(fit)) == k
    @test nobs(fit) == n

    V = vcov(fit)
    @test size(V) == (k, k)
    @test V ≈ V' rtol = 1e-10
    @test all(diag(V) .> 0)
    @test stderror(fit) ≈ sqrt.(diag(V))

    ci95 = confint(fit); ci50 = confint(fit; level = 0.5)
    @test length(ci95) == k && length(ci50) == k
    @test [c.estimate for c in ci95] ≈ coef(fit)
    for (a, b) in zip(ci95, ci50)
        @test a.lower < b.lower < b.estimate < b.upper < a.upper
    end

    @test aic(fit) ≈ -2 * loglik(fit) + 2 * k
    @test bic(fit) ≈ -2 * loglik(fit) + k * log(n)
    @test dof(fit) == k
    @test dof_residual(fit) == n - k

    pp = predict_parameters(fit, data)
    @test length(pp[:mu]) == n
    @test loglik(fit) ≈ ll_ref(pp, y) rtol = 1e-8

    r = residuals(fit)
    @test length(r) == n && all(isfinite, r)
    rq = residuals(fit; type = :quantile)
    @test length(rq) == n && all(isfinite, rq)
    @test abs(mean(rq)) < 0.25 && 0.7 < std(rq) < 1.3
    @test_throws ArgumentError residuals(fit; type = :pearson)

    s1 = simulate(fit; rng = MersenneTwister(5))
    s2 = simulate(fit; rng = MersenneTwister(5))
    s3 = simulate(fit; rng = MersenneTwister(6))
    @test length(s1) == n
    @test s1 == s2
    @test s1 != s3
    @test all(support, s1)
    return pp
end

@testset "count / LogNormal contract" begin
    # explicit generators: nested testsets share the parent's scoped RNG stream, so a
    # bare randn(n) inside one would reproduce `x`
    n = 300
    rng = MersenneTwister(20261020)
    x = randn(rng, n)
    ημ = 0.8 .+ 0.3 .* x
    f = bf(@formula(y ~ x))

    @testset "LogNormal()" begin
        y = exp.(0.5 .+ 0.3 .* x .+ 0.4 .* randn(rng, n))
        data = (; y, x)
        fit = drm(f, LogNormal(); data = data)
        pp = _contract_checks(fit, data, y, 3,
            (pp, y) -> sum(Distributions.logpdf.(
                Distributions.LogNormal.(log.(pp[:mu]), pp[:sigma]), y));
            support = >(0))
        @test coef(fit, :mu) ≈ [0.5, 0.3] atol = 0.15
        @test exp(coef(fit, :sigma)[1]) ≈ 0.4 atol = 0.08

        # response scale is exp(η); link scale is η; σ is exp(ησ)
        pl = predict_parameters(fit, data; type = :link)
        @test pl[:mu] ≈ log.(pp[:mu])
        @test pp[:mu] ≈ exp.(coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* x)
        @test all(pl[:sigma] .≈ coef(fit, :sigma)[1])
        @test pp[:sigma] ≈ exp.(pl[:sigma])
        @test fitted(fit) ≈ pp[:mu]
        @test predict(fit, data) ≈ fitted(fit)
        @test residuals(fit) ≈ y .- fitted(fit)

        @test_throws ErrorException("LogNormal() requires strictly positive responses") drm(
            f, LogNormal(); data = (; y = vcat(-1.0, y[2:end]), x))
        @test_throws ErrorException("LogNormal() requires strictly positive responses") drm(
            f, LogNormal(); data = (; y = vcat(0.0, y[2:end]), x))
        # NaN is not refused: the row is dropped with a count (#1089)
        fitN = @test_logs (:warn, r"1 of 300 rows have a missing/NaN response") drm(
            f, LogNormal(); data = (; y = vcat(NaN, y[2:end]), x))
        @test nobs(fitN) == n - 1
    end

    @testset "Poisson()" begin
        y = [rand(rng, Distributions.Poisson(exp(η))) for η in ημ]
        data = (; y, x)
        fit = drm(f, Poisson(); data = data)
        pp = _contract_checks(fit, data, y, 2,
            (pp, y) -> sum(Distributions.logpdf.(Distributions.Poisson.(pp[:mu]), y));
            support = v -> v >= 0 && v == round(v))
        @test coef(fit, :mu) ≈ [0.8, 0.3] atol = 0.15

        pl = predict_parameters(fit, data; type = :link)
        @test haskey(pl, :mu) && !haskey(pl, :sigma)
        @test pp[:mu] ≈ exp.(pl[:mu])
        @test pl[:mu] ≈ coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* x
        @test fitted(fit) ≈ pp[:mu]
        @test predict(fit, data) ≈ fitted(fit)
        @test predict(fit, data; type = :link) ≈ pl[:mu]
        @test residuals(fit) ≈ y .- fitted(fit)
        @test mean(fitted(fit)) ≈ mean(y) rtol = 1e-3   # score equation, log link + intercept

        for bad in (-1.0, 1.5)
            @test_throws ErrorException("Poisson() requires non-negative integer counts as the response") drm(
                f, Poisson(); data = (; y = vcat(bad, y[2:end]), x))
        end
        fitN = @test_logs (:warn, r"1 of 300 rows have a missing/NaN response") drm(
            f, Poisson(); data = (; y = vcat(NaN, y[2:end]), x))
        @test nobs(fitN) == n - 1
    end

    @testset "NegBinomial2()" begin
        θ = 2.0   # size; σ = 1/√θ in the twin convention
        y = [rand(rng, Distributions.NegativeBinomial(θ, θ / (θ + exp(η)))) for η in ημ]
        data = (; y, x)
        fit = drm(f, NegBinomial2(); data = data)
        pp = _contract_checks(fit, data, y, 3,
            (pp, y) -> sum(Distributions.logpdf(
                Distributions.NegativeBinomial(1 / s^2, (1 / s^2) / (1 / s^2 + m)), yi)
                for (m, s, yi) in zip(pp[:mu], pp[:sigma], y));
            support = v -> v >= 0 && v == round(v))
        @test coef(fit, :mu) ≈ [0.8, 0.3] atol = 0.2

        pl = predict_parameters(fit, data; type = :link)
        @test pp[:mu] ≈ exp.(pl[:mu])
        @test pp[:sigma] ≈ exp.(pl[:sigma])
        @test all(pl[:sigma] .≈ coef(fit, :sigma)[1])
        @test pl[:mu] ≈ coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* x
        @test fitted(fit) ≈ pp[:mu]
        @test predict(fit, data) ≈ fitted(fit)
        @test residuals(fit) ≈ y .- fitted(fit)

        for bad in (-1.0, 1.5)
            @test_throws ErrorException("NegBinomial2() requires non-negative integer counts as the response") drm(
                f, NegBinomial2(); data = (; y = vcat(bad, y[2:end]), x))
        end
        fitN = @test_logs (:warn, r"1 of 300 rows have a missing/NaN response") drm(
            f, NegBinomial2(); data = (; y = vcat(NaN, y[2:end]), x))
        @test nobs(fitN) == n - 1
    end
end
