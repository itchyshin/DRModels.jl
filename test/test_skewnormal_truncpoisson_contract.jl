# Accessor contract for SkewNormal() and TruncatedPoisson() (no hu). Test-only:
# pins what predict_parameters / fitted / residuals / vcov / confint / simulate /
# loglik do today, so a later change to either family cannot move them silently.
# The family-specific recovery and refusal tests live in test_skewnormal.jl and
# test_truncated_poisson.jl; this file does not repeat them, and says nothing
# about the nu-near-0 Wald warning (#1060) or is_converged on run-off fits.
using DRModels
using Test, Random, Statistics, LinearAlgebra
import Distributions
using Distributions: SkewNormal as DSkewNormal

# Moment (μ = mean, σ = SD, α = slant) -> Azzalini (ξ, ω, α); same map as the
# generator in test_skewnormal.jl.
function _sn_contract_internal(μ, σ, α)
    δ = α / sqrt(1 + α^2)
    ω = σ / sqrt(1 - 2 * δ^2 / π)
    return μ - ω * δ * sqrt(2 / π), ω
end

@testset "SkewNormal() accessor contract" begin
    Random.seed!(20261019)
    n = 1500
    x = randn(n)
    ξ, ω = _sn_contract_internal(1.0, 1.5, 3.0)
    y = [rand(DSkewNormal(ξ, ω, 3.0)) for _ in 1:n]
    data = (; y, x)
    form = bf(@formula(y ~ 1), @formula(sigma ~ 1), @formula(nu ~ 1))
    fit = drm(form, SkewNormal(); data = data)
    @test fit.converged

    μ̂ = coef(fit, :mu)[1]
    σ̂ = exp(coef(fit, :sigma)[1])
    α̂ = coef(fit, :nu)[1]

    @testset "predict_parameters: mu and nu identity, sigma positive" begin
        pp = predict_parameters(fit, data)
        @test sort(collect(keys(pp))) == [:mu, :nu, :sigma]
        @test all(pp[:mu] .== μ̂)
        @test all(>(0), pp[:sigma])
        @test all(pp[:sigma] .≈ σ̂)
        # module docs: nu is the slant itself, NOT exp(eta)
        @test all(pp[:nu] .== α̂)
        @test pp[:nu][1] != exp(α̂)
    end

    @testset "fitted is the mean mu (moment parameterisation)" begin
        # public mu is already the MEAN of y, so no skew shift is added on top
        # (mu + sigma*delta*sqrt(2/pi) would double count it).
        @test fitted(fit) == fill(μ̂, n)
        δ = α̂ / sqrt(1 + α̂^2)
        @test !isapprox(fitted(fit)[1], μ̂ + σ̂ * δ * sqrt(2 / π); atol = 1e-3)
    end

    @testset "residuals" begin
        r = residuals(fit)
        @test length(r) == n && all(isfinite, r)
        @test r ≈ y .- μ̂
        # documented: no verified CDF mapping, so quantile residuals are refused
        @test_throws ArgumentError residuals(fit; type = :quantile)
    end

    @testset "vcov / stderror / confint" begin
        V = vcov(fit)
        @test size(V) == (3, 3)
        @test issymmetric(V)
        @test all(diag(V) .> 0)
        @test stderror(fit) ≈ sqrt.(diag(V))
        ci = confint(fit)
        @test length(ci) == 3
        for row in ci
            @test row.lower < row.estimate < row.upper
        end
        for p in (:mu, :sigma)
            row = only(filter(r -> r.param === p, ci))
            @test row.lower < coef(fit, p)[1] < row.upper
        end
    end

    @testset "simulate matches the fitted moments" begin
        ys = simulate(fit; rng = Random.MersenneTwister(7))
        @test length(ys) == n && all(isfinite, ys)
        @test mean(ys) ≈ μ̂ rtol = 0.05
        @test std(ys) ≈ σ̂ rtol = 0.05
    end

    @testset "loglik is the closed-form skew-normal density" begin
        ξ̂, ω̂ = _sn_contract_internal(μ̂, σ̂, α̂)
        z = (y .- ξ̂) ./ ω̂
        N = Distributions.Normal()
        ll = sum(log(2 / ω̂) + Distributions.logpdf(N, zi) + Distributions.logcdf(N, α̂ * zi) for zi in z)
        @test loglik(fit) ≈ ll atol = 1e-8
        @test sum(Distributions.logpdf(DSkewNormal(ξ̂, ω̂, α̂), yi) for yi in y) ≈ loglik(fit) atol = 1e-8
    end

    @testset "non-finite / missing responses" begin
        yb = Vector{Union{Missing,Float64}}(y)
        # Inf is rejected by the shared finite-input guard before optimisation (#1123).
        yInf = copy(yb); yInf[1] = Inf
        @test_throws ArgumentError drm(form, SkewNormal(); data = (; y = yInf, x))
        msg = try; drm(form, SkewNormal(); data = (; y = yInf, x)); ""; catch e; sprint(showerror, e); end
        @test occursin("response `y`", msg)
        @test occursin("Inf is not a missing value", msg)
        # missing / NaN are already dropped with a counted warning (#1089); the
        # dropped row stays in the full-length outputs but not in nobs
        ym = copy(yb); ym[1] = missing
        fm = @test_logs (:warn, r"missing/NaN response") drm(form, SkewNormal(); data = (; y = ym, x))
        @test nobs(fm) == n - 1
        @test length(fitted(fm)) == n
        @test isfinite(loglik(fm))
    end
end

@testset "TruncatedPoisson() (no hu) accessor contract" begin
    Random.seed!(20261020)
    n = 1500
    x = randn(n)
    rtp(λ) = (while true; k = rand(Distributions.Poisson(λ)); k > 0 && return k; end)
    y = Float64[rtp(exp(0.5 + 0.3 * x[i])) for i in 1:n]
    data = (; y, x)
    fit = drm(bf(@formula(y ~ x)), TruncatedPoisson(); data = data)
    @test fit.converged
    β̂ = coef(fit, :mu)
    λ = exp.(β̂[1] .+ β̂[2] .* x)

    @testset "loglik is the zero-truncated Poisson density" begin
        ll = sum(Distributions.logpdf(Distributions.Poisson(λ[i]), y[i]) - log1p(-exp(-λ[i])) for i in 1:n)
        @test loglik(fit) ≈ ll atol = 1e-8
    end

    @testset "predict_parameters / fitted are the untruncated Poisson mean" begin
        pp = predict_parameters(fit, data)
        @test collect(keys(pp)) == [:mu]
        @test pp[:mu] ≈ λ
        @test all(>(0), pp[:mu])
        # fitted() reports the Poisson parameter lambda, NOT the truncated mean
        # lambda/(1 - exp(-lambda)); pinned so a change to either shows up here.
        @test fitted(fit) ≈ λ
        @test !isapprox(fitted(fit), λ ./ (1 .- exp.(-λ)); rtol = 1e-3)
    end

    @testset "simulate is strictly positive with the truncated mean" begin
        ys = simulate(fit; rng = Random.MersenneTwister(3))
        @test length(ys) == n
        @test all(>=(1), ys) && all(isinteger, ys)
        @test !any(==(0), ys)
        m = λ ./ (1 .- exp.(-λ))                       # truncated mean
        v = m .* (1 .+ λ .- m)                          # truncated variance
        se = sqrt(sum(v)) / n
        @test abs(mean(ys) - mean(m)) < 4 * se
        # many replicates: tight check on the same moment
        reps = [mean(simulate(fit; rng = Random.MersenneTwister(100 + k))) for k in 1:20]
        @test mean(reps) ≈ mean(m) rtol = 0.01
    end

    @testset "refusals (new cases)" begin
        msg(f) = try; f(); ""; catch e; sprint(showerror, e); end
        for bad in (-3.0, 2.5)
            yb = copy(y); yb[1] = bad
            err = msg(() -> drm(bf(@formula(y ~ x)), TruncatedPoisson(); data = (; y = yb, x)))
            @test occursin("requires positive integer counts", err)
        end
        yb = copy(y); yb[1] = 0.0
        @test occursin("requires positive integer counts",
                       msg(() -> drm(bf(@formula(y ~ x)), TruncatedPoisson(); data = (; y = yb, x))))
    end

    @testset "vcov / confint / aic / bic" begin
        V = vcov(fit)
        @test size(V) == (2, 2) && issymmetric(V) && all(diag(V) .> 0)
        @test stderror(fit) ≈ sqrt.(diag(V))
        ci = confint(fit)
        @test length(ci) == 2
        for (j, row) in enumerate(ci)
            @test row.estimate == β̂[j]
            @test row.lower < row.estimate < row.upper
        end
        # a 95% interval misses the truth about 1 draw in 20 and the draw differs by
        # Julia version, so check the slope is within 4 SE of the truth instead
        @test abs(ci[2].estimate - 0.3) < 4 * stderror(fit)[2]
        k = length(β̂)
        @test aic(fit) ≈ -2 * loglik(fit) + 2k
        @test bic(fit) ≈ -2 * loglik(fit) + k * log(n)
    end

    @testset "residuals finite" begin
        @test all(isfinite, residuals(fit))
        rq = residuals(fit; type = :quantile, rng = Random.MersenneTwister(5))
        @test length(rq) == n && all(isfinite, rq)
    end
end
