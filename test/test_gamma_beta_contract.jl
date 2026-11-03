# Accessor / refusal contract for the fixed-effect Gamma() and Beta() families.
# Test-only: pins what drm / fitted / predict_parameters / residuals / vcov /
# confint / aic / bic / simulate do today on small seeded fits, so a later change
# to either family cannot move them silently. Recovery lives in test_beta.jl and
# test_164_gamma_hetero.jl; random-effect routes are out of scope here (Beta
# (1 | g) marginal = :Laplace #1093, crossed #1142 and location-scale
# convergence #1127 are tracked elsewhere and deliberately not touched).
using DRModels
using Test, Random, Statistics, LinearAlgebra
import Distributions
using StableRNGs: StableRNG

_msg_gb(f) = try; f(); ""; catch e; sprint(showerror, e); end

# Standard-normal-scale quantile residuals should be ~ N(0,1), i.e. the PIT
# Φ(r) roughly uniform: check the decile counts and the first two moments.
function _pit_roughly_uniform(r)
    u = Distributions.cdf.(Distributions.Normal(), r)
    n = length(u)
    @test abs(mean(u) - 0.5) < 0.04
    @test abs(var(u) - 1 / 12) < 0.02
    counts = [count(x -> (k - 1) / 10 <= x < k / 10, u) for k in 1:10]
    @test all(abs.(counts ./ n .- 0.1) .< 0.04)
end

@testset "Gamma() fixed-effect contract" begin
    rng = StableRNG(20261020)
    n = 1500
    x = randn(rng, n)
    α = 10.0                                   # shape; sigma = 1/sqrt(alpha)
    μ = exp.(0.5 .+ 0.3 .* x)
    y = [rand(rng, Distributions.Gamma(α, μ[i] / α)) for i in 1:n]
    data = (; y, x)
    form = bf(@formula(y ~ x), @formula(sigma ~ 1))
    fit = drm(form, Gamma(); data = data)
    k = length(coef(fit))

    @testset "refusals: out-of-support responses" begin
        for bad in (0.0, -1.0)
            yb = copy(y); yb[1] = bad
            err = _msg_gb(() -> drm(form, Gamma(); data = (; y = yb, x)))
            @test occursin("Gamma() requires strictly positive responses", err)
        end
        @test_throws ErrorException drm(form, Gamma(); data = (; y = vcat(0.0, y[2:end]), x))
    end

    @testset "fit shape" begin
        @test k == 3
        @test is_converged(fit)
        @test coef(fit, :mu)[2] ≈ 0.3 atol = 0.05
        @test coef(fit, :mu)[1] ≈ 0.5 atol = 0.05
    end

    @testset "fitted is the response-scale mean exp(Xβ)" begin
        β̂ = coef(fit, :mu)
        ŷ = fitted(fit)
        @test length(ŷ) == n
        @test ŷ ≈ exp.(β̂[1] .+ β̂[2] .* x)
        @test all(>(0), ŷ)
        @test predict(fit, data) ≈ ŷ
    end

    @testset "predict_parameters: mu on the response scale, sigma = 1/sqrt(alpha)" begin
        pp = predict_parameters(fit, data)
        @test sort(collect(keys(pp))) == [:mu, :sigma]
        @test pp[:mu] ≈ fitted(fit)
        @test all(>(0), pp[:sigma])
        @test all(pp[:sigma] .≈ exp(coef(fit, :sigma)[1]))
        @test pp[:sigma][1] ≈ 1 / sqrt(α) rtol = 0.1      # true sigma = 0.316
        @test sigma(fit) ≈ pp[:sigma]
    end

    @testset "residuals" begin
        r = residuals(fit)
        @test length(r) == n && all(isfinite, r)
        @test r ≈ y .- fitted(fit)
        rq = residuals(fit; type = :quantile, rng = StableRNG(1))
        @test length(rq) == n && all(isfinite, rq)
        _pit_roughly_uniform(rq)
    end

    @testset "vcov / stderror / confint" begin
        V = vcov(fit)
        @test size(V) == (k, k)
        @test V ≈ V' rtol = 1e-10
        @test all(diag(V) .> 0)
        @test stderror(fit) ≈ sqrt.(diag(V))
        ci = confint(fit)
        @test length(ci) == k
        for (j, row) in enumerate(ci)
            @test row.estimate == coef(fit)[j]
            @test row.lower < row.estimate < row.upper
        end
        ci90, ci99 = confint(fit; level = 0.90), confint(fit; level = 0.99)
        for j in 1:k
            @test ci90[j].lower > ci[j].lower && ci90[j].upper < ci[j].upper
            @test ci[j].lower > ci99[j].lower && ci[j].upper < ci99[j].upper
        end
        @test ci[2].lower < 0.3 < ci[2].upper
    end

    @testset "aic / bic arithmetic" begin
        @test dof(fit) == k
        @test nobs(fit) == n
        @test aic(fit) ≈ -2 * loglik(fit) + 2k
        @test bic(fit) ≈ -2 * loglik(fit) + k * log(n)
    end

    @testset "loglik is the Gamma(alpha, mu/alpha) density" begin
        σ̂ = exp(coef(fit, :sigma)[1]); α̂ = 1 / σ̂^2
        ll = sum(Distributions.logpdf(Distributions.Gamma(α̂, fitted(fit)[i] / α̂), y[i]) for i in 1:n)
        @test loglik(fit) ≈ ll atol = 1e-6
    end

    @testset "simulate: support and seed reproducibility" begin
        s1 = simulate(fit; rng = StableRNG(5))
        s2 = simulate(fit; rng = StableRNG(5))
        s3 = simulate(fit; rng = StableRNG(6))
        @test length(s1) == n
        @test all(>(0), s1) && all(isfinite, s1)
        @test s1 == s2
        @test s1 != s3
        @test mean(s1) ≈ mean(y) rtol = 0.1
    end

    # Issue #992 / #959 document a dispersion cap of exp(8) = 2981 (Gamma CV
    # below 1.8%) on the crossed / structured Laplace routes. The fixed-effect
    # route is NOT capped there: `_fit_gamma` only soft-clamps eta_sigma at +-15
    # (shape up to exp(30)). Pin current behaviour: a CV ~ 1.4% fit (shape 5000,
    # past exp(8)) recovers a shape above the exp(8) cap and reports converged.
    @testset "dispersion cap at exp(8) is not applied on the fixed-effect route (#992)" begin
        rl = StableRNG(99)
        yl = [rand(rl, Distributions.Gamma(5000.0, μ[i] / 5000.0)) for i in 1:n]
        fl = drm(form, Gamma(); data = (; y = yl, x))
        shape = exp(-2 * coef(fl, :sigma)[1])
        @test is_converged(fl)
        @test shape > exp(8)
        @test shape ≈ 5000 rtol = 0.2
    end
end

@testset "Beta() fixed-effect contract" begin
    rng = StableRNG(20261021)
    n = 1500
    x = randn(rng, n)
    φ = 15.0                                   # precision; sigma = 1/sqrt(phi)
    μ = 1 ./ (1 .+ exp.(-(0.3 .+ 0.8 .* x)))
    y = [rand(rng, Distributions.Beta(μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n]
    data = (; y, x)
    form = bf(@formula(y ~ x), @formula(sigma ~ 1))
    fit = drm(form, Beta(); data = data)
    k = length(coef(fit))

    @testset "refusals: out-of-support responses" begin
        for bad in (0.0, 1.0, -0.1, 1.2)
            yb = copy(y); yb[1] = bad
            err = _msg_gb(() -> drm(form, Beta(); data = (; y = yb, x)))
            @test occursin("Beta() requires responses strictly in the open interval (0, 1)", err)
        end
        # The refusal for exact 0 / 1 should point the user at ZeroOneBeta().
        # It does not today: the message is the generic support error.
        for bad in (0.0, 1.0)
            yb = copy(y); yb[1] = bad
            err = _msg_gb(() -> drm(form, Beta(); data = (; y = yb, x)))
            @test_broken occursin("ZeroOneBeta", err)
        end
    end

    @testset "fit shape" begin
        @test k == 3
        @test is_converged(fit)
        @test coef(fit, :mu)[1] ≈ 0.3 atol = 0.08
        @test coef(fit, :mu)[2] ≈ 0.8 atol = 0.08
    end

    @testset "fitted is the response-scale mean logistic(Xβ)" begin
        β̂ = coef(fit, :mu)
        ŷ = fitted(fit)
        @test length(ŷ) == n
        @test ŷ ≈ 1 ./ (1 .+ exp.(-(β̂[1] .+ β̂[2] .* x)))
        @test all(0 .< ŷ .< 1)
        @test predict(fit, data) ≈ ŷ
    end

    @testset "predict_parameters: mu on the response scale, sigma = 1/sqrt(phi)" begin
        pp = predict_parameters(fit, data)
        @test sort(collect(keys(pp))) == [:mu, :sigma]
        @test pp[:mu] ≈ fitted(fit)
        @test all(0 .< pp[:mu] .< 1)
        @test all(>(0), pp[:sigma])
        @test all(pp[:sigma] .≈ exp(coef(fit, :sigma)[1]))
        @test pp[:sigma][1] ≈ 1 / sqrt(φ) rtol = 0.1      # true sigma = 0.258
        @test sigma(fit) ≈ pp[:sigma]
    end

    @testset "residuals" begin
        r = residuals(fit)
        @test length(r) == n && all(isfinite, r)
        @test r ≈ y .- fitted(fit)
        rq = residuals(fit; type = :quantile, rng = StableRNG(1))
        @test length(rq) == n && all(isfinite, rq)
        _pit_roughly_uniform(rq)
    end

    @testset "vcov / stderror / confint" begin
        V = vcov(fit)
        @test size(V) == (k, k)
        @test V ≈ V' rtol = 1e-10
        @test all(diag(V) .> 0)
        @test stderror(fit) ≈ sqrt.(diag(V))
        ci = confint(fit)
        @test length(ci) == k
        for (j, row) in enumerate(ci)
            @test row.estimate == coef(fit)[j]
            @test row.lower < row.estimate < row.upper
        end
        ci90, ci99 = confint(fit; level = 0.90), confint(fit; level = 0.99)
        for j in 1:k
            @test ci90[j].lower > ci[j].lower && ci90[j].upper < ci[j].upper
            @test ci[j].lower > ci99[j].lower && ci[j].upper < ci99[j].upper
        end
        @test ci[2].lower < 0.8 < ci[2].upper
    end

    @testset "aic / bic arithmetic" begin
        @test dof(fit) == k
        @test nobs(fit) == n
        @test aic(fit) ≈ -2 * loglik(fit) + 2k
        @test bic(fit) ≈ -2 * loglik(fit) + k * log(n)
    end

    @testset "loglik is the Beta(mu*phi, (1-mu)*phi) density" begin
        φ̂ = exp(-2 * coef(fit, :sigma)[1])
        m = fitted(fit)
        ll = sum(Distributions.logpdf(Distributions.Beta(m[i] * φ̂, (1 - m[i]) * φ̂), y[i]) for i in 1:n)
        @test loglik(fit) ≈ ll atol = 1e-6
    end

    @testset "simulate: support and seed reproducibility" begin
        s1 = simulate(fit; rng = StableRNG(5))
        s2 = simulate(fit; rng = StableRNG(5))
        s3 = simulate(fit; rng = StableRNG(6))
        @test length(s1) == n
        @test all(0 .< s1 .< 1)
        @test s1 == s2
        @test s1 != s3
        @test mean(s1) ≈ mean(y) atol = 0.02
    end

    # Same note as Gamma (#992 / #959): the exp(8) precision cap belongs to the
    # crossed / structured routes and zeroonebeta.jl. The fixed-effect `_fit_beta`
    # only soft-clamps eta_sigma at +-15, so a tight-proportion fit (phi = 1e5,
    # far past exp(8) = 2981) is not pinned at the cap. Pinned as-is.
    @testset "precision cap at exp(8) is not applied on the fixed-effect route (#992)" begin
        rl = StableRNG(98)
        yl = [rand(rl, Distributions.Beta(μ[i] * 1e5, (1 - μ[i]) * 1e5)) for i in 1:n]
        fl = drm(form, Beta(); data = (; y = yl, x))
        prec = exp(-2 * coef(fl, :sigma)[1])
        @test is_converged(fl)
        @test prec > exp(8)
    end
end
