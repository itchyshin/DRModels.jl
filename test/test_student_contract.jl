# Accessor contract for the fixed-effect Student() family. Test-only: pins what
# predict_parameters / fitted / residuals / vcov / confint / aic / simulate /
# coeftable and the input refusals do today, so a later change cannot move them
# silently. Recovery lives in test_student.jl; random-effect routes in
# test_student_re.jl and friends. This file says nothing about nu at the
# Gaussian limit (#937 #961 #969) nor about Student with (1 | g).
using DRModels
using Test, LinearAlgebra, Statistics
using StableRNGs
using Distributions: TDist, Normal, cdf, quantile

@testset "Student() accessor contract (fixed effects)" begin
    rng = StableRNG(20261020)
    n = 600
    x = randn(rng, n)
    y = 0.5 .+ 0.4 .* x .+ 0.8 .* rand(rng, TDist(5), n)
    data = (; y, x)
    form = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(nu ~ 1))
    fit = drm(form, Student(); data = data)
    k = 4
    @test fit.converged
    @test length(coef(fit)) == k

    @testset "predict_parameters: mu identity, sigma = exp(eta), nu = 2 + exp(eta)" begin
        pp = predict_parameters(fit, data)
        @test sort(collect(keys(pp))) == [:mu, :nu, :sigma]
        X = hcat(ones(n), x)
        @test pp[:mu] ≈ X * coef(fit, :mu)
        @test all(pp[:sigma] .≈ exp(coef(fit, :sigma)[1]))
        @test all(pp[:nu] .≈ 2 + exp(coef(fit, :nu)[1]))
        @test all(>(2), pp[:nu])           # finite variance by construction
        @test all(>(0), pp[:sigma])
    end

    @testset "nu formula with a covariate varies by row" begin
        f2 = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(nu ~ 1 + x))
        fit2 = drm(f2, Student(); data = data)
        @test length(coef(fit2)) == k + 1
        nu2 = predict_parameters(fit2, data)[:nu]
        @test all(>(2), nu2)
        @test nu2 ≈ 2 .+ exp.(coef(fit2, :nu)[1] .+ coef(fit2, :nu)[2] .* x)
        @test length(unique(round.(nu2; digits = 8))) > 1
        # nu omitted from the bf() is the constant-nu model
        fit3 = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Student(); data = data)
        @test coef(fit3) ≈ coef(fit)
        @test loglik(fit3) ≈ loglik(fit)
    end

    @testset "fitted is the location mu; residuals" begin
        ŷ = fitted(fit)
        @test length(ŷ) == n
        @test ŷ ≈ predict_parameters(fit, data)[:mu]
        @test predict(fit, data) ≈ ŷ
        r = residuals(fit)
        @test r ≈ y .- ŷ
        q = residuals(fit; type = :quantile)
        @test length(q) == n && all(isfinite, q)
        σ̂ = exp(coef(fit, :sigma)[1]); ν̂ = 2 + exp(coef(fit, :nu)[1])
        pit = cdf.(TDist(ν̂), (y .- ŷ) ./ σ̂)
        @test q ≈ quantile.(Normal(), pit) atol = 1e-6
        @test abs(mean(q)) < 0.15
        @test 0.85 < std(q) < 1.15
    end

    @testset "vcov / stderror / confint / coeftable" begin
        V = vcov(fit)
        @test size(V) == (k, k)
        @test issymmetric(V)
        @test minimum(eigvals(Symmetric(V))) > -1e-10        # PSD
        @test all(diag(V) .> 0)
        @test stderror(fit) ≈ sqrt.(diag(V))

        ci95 = confint(fit)
        ci90 = confint(fit; level = 0.9)
        @test length(ci95) == k && length(ci90) == k
        @test [c.estimate for c in ci95] ≈ coef(fit)
        for (a, b) in zip(ci95, ci90)
            @test a.lower < b.lower < b.estimate < b.upper < a.upper   # 0.9 nests in 0.95
        end

        ct = coeftable(fit)
        @test ct.rownms == ["mu: (Intercept)", "mu: x", "sigma: (Intercept)", "nu: (Intercept)"]
        @test ct.colnms == ["Estimate", "Std.Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"]
        @test ct.cols[1] ≈ coef(fit)
        @test ct.cols[2] ≈ stderror(fit)
    end

    @testset "loglik / aic / bic arithmetic" begin
        @test nobs(fit) == n
        @test dof(fit) == k
        @test aic(fit) ≈ -2 * loglik(fit) + 2 * k
        @test bic(fit) ≈ -2 * loglik(fit) + k * log(n)
        @test bic(fit) > aic(fit)
        @test isfinite(loglik(fit))
    end

    @testset "simulate: shape and seed reproducibility" begin
        s1 = simulate(fit; rng = StableRNG(1))
        s2 = simulate(fit; rng = StableRNG(1))
        s3 = simulate(fit; rng = StableRNG(2))
        @test length(s1) == n && all(isfinite, s1)
        @test s1 == s2
        @test s1 != s3
        @test mean(s1) ≈ mean(fitted(fit)) atol = 0.2
    end

    @testset "refits are reproducible" begin
        fitb = drm(form, Student(); data = data)
        @test coef(fitb) == coef(fit)
        @test loglik(fitb) == loglik(fit)
    end

    @testset "refusals" begin
        # nu = 2 + exp(eta) cannot leave (2, Inf), so there is no invalid-nu input to refuse.
        @test_throws ErrorException("Student() does not support meta_V / structured markers") drm(
            bf(@formula(y ~ x + meta_V(x))), Student(); data = data)
        g = repeat(1:6, inner = 100)
        @test_throws ErrorException("Student(): only the mean formula may carry a random effect") drm(
            bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g))), Student(); data = (; y, x, g))
        @test_throws ArgumentError("drm (Student): marginal = :foo is not available; use `:LA` (default) " *
                                   "or `:Laplace` (ordinary `(1 | g)` on the mean).") drm(
            form, Student(); data = data, marginal = :foo)
        @test_throws r"marginal = :Laplace is not available for Student\(\) with no random effect" drm(
            form, Student(); data = data, marginal = :Laplace)

        # tracks #1123: an Inf response reaches the optimiser and throws a
        # linear-algebra ArgumentError rather than a named refusal. Pinned, not fixed.
        yInf = copy(y); yInf[1] = Inf
        @test_throws ArgumentError drm(form, Student(); data = (; y = yInf, x))
        msg = try; drm(form, Student(); data = (; y = yInf, x)); ""; catch e; sprint(showerror, e); end
        @test occursin("matrix contains Infs or NaNs", msg)

        # NaN responses are dropped with a counted warning (#1089), not refused;
        yNaN = copy(y); yNaN[1] = NaN
        fm = @test_logs (:warn, r"missing/NaN response") drm(form, Student(); data = (; y = yNaN, x))
        # the dropped row stays in the full-length outputs (NaN residual) but not in nobs
        @test nobs(fm) == n - 1
        @test length(fitted(fm)) == n
        @test isnan(residuals(fm)[1]) && all(isfinite, residuals(fm)[2:end])
        @test isfinite(loglik(fm))
    end
end
