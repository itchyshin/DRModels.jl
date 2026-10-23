# Accessor contract for the fixed-effect BetaBinomial() (cbind(successes, failures))
# and TruncatedNegBinomial2() (no hu). Test-only: pins what the refusals,
# predict_parameters / fitted / residuals / vcov / confint / aic / bic / simulate /
# loglik do today, so a later change to either family cannot move them silently.
# Family recovery lives in test_betabinomial.jl and test_truncated_nb.jl; this file
# does not repeat it. Random effects are out of scope (TruncatedNegBinomial2 with
# (1 | g) is refused, tracked in #1122, and is deliberately not exercised here).
using DRModels
using Test, Random, Statistics, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma, logbeta

_bbtn_msg(f) = try; f(); ""; catch e; sprint(showerror, e); end

@testset "BetaBinomial() accessor contract" begin
    Random.seed!(20261021)
    N = 800
    x = randn(N)
    ntr = rand(10:30, N)
    μ0 = 1 ./ (1 .+ exp.(-(0.2 .+ 0.6 .* x)))
    φ0 = 10.0
    s = [rand(Distributions.BetaBinomial(ntr[i], μ0[i] * φ0, (1 - μ0[i]) * φ0)) for i in 1:N]
    fl = ntr .- s
    data = (; s = Float64.(s), fl = Float64.(fl), x)
    form = bf(@formula(cbind(s, fl) ~ x), @formula(sigma ~ 1))
    fit = drm(form, BetaBinomial(); data = data)
    @test fit.converged

    βμ = coef(fit, :mu)
    σ̂ = exp(coef(fit, :sigma)[1])
    φ̂ = 1 / σ̂^2
    μ̂ = 1 ./ (1 .+ exp.(-(βμ[1] .+ βμ[2] .* x)))

    @testset "refusals" begin
        refuse(sv, fv) = _bbtn_msg(() -> drm(form, BetaBinomial(); data = (; s = sv, fl = fv, x)))
        nonint = copy(data.s); nonint[1] = 2.5
        @test occursin("non-negative integer", refuse(nonint, data.fl))
        @test occursin("non-negative integer", refuse(data.s, replace(data.fl, data.fl[1] => 3.5)))
        neg = copy(data.s); neg[1] = -1.0
        @test occursin("non-negative integer", refuse(neg, data.fl))
        negf = copy(data.fl); negf[2] = -2.0
        @test occursin("non-negative integer", refuse(data.s, negf))
        # a row with zero trials (0 successes, 0 failures) should be refused; today
        # it is accepted silently and contributes 0 to the log-likelihood.
        s0 = copy(data.s); f0 = copy(data.fl); s0[1] = 0.0; f0[1] = 0.0
        @test_broken !isempty(_bbtn_msg(() -> drm(form, BetaBinomial(); data = (; s = s0, fl = f0, x))))
        # a one-column response is refused
        @test occursin("two-column response",
                       _bbtn_msg(() -> drm(bf(@formula(s ~ x), @formula(sigma ~ 1)), BetaBinomial(); data = data)))
    end

    @testset "predict_parameters" begin
        pp = predict_parameters(fit, data)
        @test sort(collect(keys(pp))) == [:mu, :sigma]
        @test pp[:mu] ≈ μ̂
        @test all(0 .< pp[:mu] .< 1)
        @test all(pp[:sigma] .≈ σ̂)
        @test all(1 ./ pp[:sigma] .^ 2 .≈ φ̂)           # precision phi = 1/sigma^2
    end

    @testset "fitted is the mean success probability" begin
        @test fitted(fit) ≈ μ̂
        @test all(0 .< fitted(fit) .< 1)
    end

    @testset "residuals" begin
        r = residuals(fit)
        @test length(r) == N && all(isfinite, r)
        rq = residuals(fit; type = :quantile, rng = Random.MersenneTwister(5))
        @test length(rq) == N && all(isfinite, rq)
        @test abs(mean(rq)) < 0.2
        @test 0.8 < std(rq) < 1.2
    end

    @testset "vcov / stderror / confint / aic / bic" begin
        V = vcov(fit)
        @test size(V) == (3, 3) && issymmetric(V) && all(diag(V) .> 0)
        @test stderror(fit) ≈ sqrt.(diag(V))
        ci = confint(fit)
        @test length(ci) == 3
        for row in ci
            @test row.lower < row.estimate < row.upper
        end
        mu_rows = filter(r -> r.param === :mu, ci)
        @test length(mu_rows) == 2
        @test mu_rows[2].lower < 0.6 < mu_rows[2].upper
        k = 3
        @test aic(fit) ≈ -2 * loglik(fit) + 2k
        @test bic(fit) ≈ -2 * loglik(fit) + k * log(N)
    end

    @testset "simulate never exceeds the trials" begin
        ys = simulate(fit; rng = Random.MersenneTwister(7))
        @test length(ys) == N
        @test all(0 .<= ys .<= ntr) && all(isinteger, ys)
        reps = [simulate(fit; rng = Random.MersenneTwister(100 + k)) for k in 1:20]
        @test mean(mean.(reps)) ≈ mean(ntr .* μ̂) rtol = 0.03
        @test all(all(0 .<= r .<= ntr) for r in reps)
    end

    @testset "loglik is the independent beta-binomial pmf sum" begin
        a = μ̂ .* φ̂; b = (1 .- μ̂) .* φ̂
        ll = sum(Distributions.logpdf(Distributions.BetaBinomial(ntr[i], a[i], b[i]), s[i]) for i in 1:N)
        @test loglik(fit) ≈ ll atol = 1e-6
        # closed form, no Distributions: log C(n,s) + logB(s+a, n-s+b) - logB(a,b)
        lc(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
        ll2 = sum(lc(ntr[i], s[i]) + logbeta(s[i] + a[i], fl[i] + b[i]) - logbeta(a[i], b[i]) for i in 1:N)
        @test loglik(fit) ≈ ll2 atol = 1e-6
    end
end

@testset "TruncatedNegBinomial2() (no hu) accessor contract" begin
    Random.seed!(20261022)
    n = 1500
    x = randn(n)
    r0 = 3.0                                            # NB2 size
    rtnb(μ) = (while true
                   k = rand(Distributions.NegativeBinomial(r0, r0 / (r0 + μ)))
                   k > 0 && return k
               end)
    y = Float64[rtnb(exp(0.6 + 0.3 * x[i])) for i in 1:n]
    data = (; y, x)
    form = bf(@formula(y ~ x), @formula(sigma ~ 1))
    fit = drm(form, TruncatedNegBinomial2(); data = data)
    @test fit.converged
    βμ = coef(fit, :mu)
    σ̂ = exp(coef(fit, :sigma)[1])
    r̂ = 1 / σ̂^2
    μ̂ = exp.(βμ[1] .+ βμ[2] .* x)

    @testset "refusals" begin
        refuse(yv) = _bbtn_msg(() -> drm(form, TruncatedNegBinomial2(); data = (; y = yv, x)))
        y0 = copy(y); y0[1] = 0.0
        @test occursin("requires positive integer counts", refuse(y0))
        yn = copy(y); yn[1] = -2.0
        @test occursin("requires positive integer counts", refuse(yn))
        yf = copy(y); yf[1] = 2.5
        @test occursin("requires positive integer counts", refuse(yf))
    end

    @testset "loglik is the zero-truncated NB2 density" begin
        ll = sum(Distributions.logpdf(Distributions.NegativeBinomial(r̂, r̂ / (r̂ + μ̂[i])), y[i]) -
                 log1p(-Distributions.pdf(Distributions.NegativeBinomial(r̂, r̂ / (r̂ + μ̂[i])), 0))
                 for i in 1:n)
        @test loglik(fit) ≈ ll atol = 1e-6
    end

    @testset "predict_parameters / fitted are the untruncated NB2 mean" begin
        pp = predict_parameters(fit, data)
        @test sort(collect(keys(pp))) == [:mu, :sigma]
        @test pp[:mu] ≈ μ̂
        @test all(>(0), pp[:mu])
        @test all(pp[:sigma] .≈ σ̂)
        @test all(1 ./ pp[:sigma] .^ 2 .≈ r̂)           # NB2 size = 1/sigma^2
        # fitted() reports the NB2 mean, NOT the truncated mean mu/(1 - P(0))
        @test fitted(fit) ≈ μ̂
        p0 = [Distributions.pdf(Distributions.NegativeBinomial(r̂, r̂ / (r̂ + μ̂[i])), 0) for i in 1:n]
        @test !isapprox(fitted(fit), μ̂ ./ (1 .- p0); rtol = 1e-3)
    end

    @testset "residuals" begin
        r = residuals(fit)
        @test length(r) == n && all(isfinite, r)
        rq = residuals(fit; type = :quantile, rng = Random.MersenneTwister(5))
        @test length(rq) == n && all(isfinite, rq)
        # the randomised zero-truncated PIT is a proper uniform on (0, 1)
        u = Distributions.cdf.(Distributions.Normal(), rq)
        @test all(0 .< u .< 1)
        @test abs(mean(rq)) < 0.15
        @test 0.85 < std(rq) < 1.15
    end

    @testset "vcov / stderror / confint / aic / bic" begin
        V = vcov(fit)
        @test size(V) == (3, 3) && issymmetric(V) && all(diag(V) .> 0)
        @test stderror(fit) ≈ sqrt.(diag(V))
        ci = confint(fit)
        @test length(ci) == 3
        for row in ci
            @test row.lower < row.estimate < row.upper
        end
        mu_rows = filter(r -> r.param === :mu, ci)
        @test mu_rows[2].lower < 0.3 < mu_rows[2].upper
        k = 3
        @test aic(fit) ≈ -2 * loglik(fit) + 2k
        @test bic(fit) ≈ -2 * loglik(fit) + k * log(n)
    end

    @testset "simulate is strictly positive" begin
        ys = simulate(fit; rng = Random.MersenneTwister(3))
        @test length(ys) == n
        @test all(>=(1), ys) && all(isinteger, ys)
        p0 = [Distributions.pdf(Distributions.NegativeBinomial(r̂, r̂ / (r̂ + μ̂[i])), 0) for i in 1:n]
        m = μ̂ ./ (1 .- p0)                              # truncated mean
        reps = [mean(simulate(fit; rng = Random.MersenneTwister(100 + k))) for k in 1:20]
        @test mean(reps) ≈ mean(m) rtol = 0.02
    end
end
