# Randomized quantile residuals (Dunn & Smyth; DHARMa / glmmTMB style) via the
# `residuals(fit; type = :quantile)` keyword. Back-compat: `type = :response`
# (the default) is byte-identical to the old `residuals(fit)`.
#
# Per family: simulate a correctly-specified dataset (seeded), fit, and assert
# the quantile residuals are ≈ N(0,1) by moments (mean≈0, var≈1) AND a seeded
# one-sample Kolmogorov–Smirnov test against the standard normal. A wrong
# per-family parameterization fails the KS / moment gate immediately.
using DRModels
using Test, Random, Statistics
import Distributions          # qualified — DRModels exports its own Poisson/Gamma families

# One-sample KS statistic D = sup_x |F_n(x) − Φ(x)| against the standard normal.
# Under H0 (residuals ~ N(0,1)) the scaled statistic √n·D follows the Kolmogorov
# distribution; √n·D < 1.7 corresponds to roughly α ≈ 0.006. A mis-parameterized
# family yields D on the order of 0.1–0.5, i.e. √n·D ≫ 1.7, so the gate is sharp.
function ks_against_normal(r)
    n = length(r)
    s = sort(r)
    Φ(x) = Distributions.cdf(Distributions.Normal(), x)
    D = 0.0
    @inbounds for i in 1:n
        F = Φ(s[i])
        D = max(D, abs(i / n - F), abs(F - (i - 1) / n))
    end
    return D
end
# Tolerant threshold on √n·D: passes well-specified data, fails a wrong CDF map.
ks_ok(r) = sqrt(length(r)) * ks_against_normal(r) < 1.7
# Moment gate.
moments_ok(r) = abs(mean(r)) < 0.15 && 0.85 < std(r) < 1.18

@testset "quantile residuals" begin
    @testset "Gaussian — back-compat + loc-scale PIT" begin
        Random.seed!(20260604)
        n = 800
        x = randn(n)
        βμ = [0.5, -0.8]; βσ = [-0.3, 0.4]   # log σ = -0.3 + 0.4x
        μ = βμ[1] .+ βμ[2] .* x
        σ = exp.(βσ[1] .+ βσ[2] .* x)
        y = μ .+ σ .* randn(n)
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = data)

        # Back-compat: default == :response, and == y - fitted.
        @test residuals(fit) == residuals(fit; type = :response)
        @test residuals(fit) ≈ (y .- fitted(fit))

        r = residuals(fit; type = :quantile)
        μ̂ = fit.means[:mu]; σ̂ = fit.scales[:sigma]
        # Gaussian PIT through Φ⁻¹(Φ((y-μ)/σ)) = (y-μ)/σ exactly.
        @test r ≈ (y .- μ̂) ./ σ̂ rtol = 1e-8
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "Poisson — randomized discrete PIT" begin
        Random.seed!(20260615)
        n = 3000
        x = randn(n)
        β = [0.3, 0.5]
        λ = exp.(β[1] .+ β[2] .* x)
        y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x)), DRModels.Poisson(); data = data)

        Random.seed!(1)
        r = residuals(fit; type = :quantile)
        @test length(r) == n
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "Student-t — continuous PIT" begin
        Random.seed!(101)
        n = 2000
        x = randn(n)
        βμ = [0.4, -0.6]; ν = 6.0; σ = 1.3
        μ = βμ[1] .+ βμ[2] .* x
        y = μ .+ σ .* Float64.([rand(Distributions.TDist(ν)) for _ in 1:n])
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
                  Student(); data = data)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "LogNormal — continuous PIT" begin
        Random.seed!(202)
        n = 2000
        x = randn(n)
        βμ = [0.2, 0.5]; σ = 0.6        # μ = mean of log y, σ = SD of log y
        meanlog = βμ[1] .+ βμ[2] .* x
        y = exp.(meanlog .+ σ .* randn(n))
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), DRModels.LogNormal(); data = data)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "Gamma — continuous PIT (shape α = σ⁻²)" begin
        Random.seed!(303)
        n = 2000
        x = randn(n)
        β = [0.5, 0.4]; α = 8.0          # σ = 1/√α
        μ = exp.(β[1] .+ β[2] .* x)
        y = Float64.([rand(Distributions.Gamma(α, μi / α)) for μi in μ])
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), DRModels.Gamma(); data = data)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "Gamma location–scale — sigma slot holds the SHAPE (#302)" begin
        # A coupled `(1 | p | group)` triggers the location–scale Gamma route,
        # whose kernel sets shape α = exp ψ and stores scales[:sigma] = α (the
        # shape ITSELF), unlike the plain Gamma fit that stores scales[:sigma] = σ
        # with α = σ⁻². Before the fix, _conditional_dist(::Gamma) always computed
        # α = σ⁻², so for a location–scale fit it used α = 1/α² — inverting the
        # shape and producing grossly non-normal quantile residuals even for a
        # correctly-specified model (mean ≈ 1.26, std ≈ 0.09, √n·D ≈ 46). Keep the
        # group-level RE variance small so the marginal (RE = 0) predictors the
        # residual map uses are close to the per-observation truth; this isolates
        # the shape-convention defect rather than RE-marginalization slack.
        Random.seed!(424242)
        G = 60; m = 50; n = G * m
        species = repeat(1:G, inner = m)
        x = randn(n)
        βμ = [0.3, 0.4]; βψ = [1.6]              # shape α = exp ψ ≈ 4.95
        LΛ = [sqrt(0.02) 0.0; 0.0 sqrt(0.01)]    # small coupled-RE (mean, log-shape)
        A = [LΛ * randn(2) for _ in 1:G]
        y = [ (α = exp(βψ[1] + A[species[i]][2]);
               μ = exp(βμ[1] + βμ[2] * x[i] + A[species[i]][1]);
               rand(Distributions.Gamma(α, μ / α))) for i in 1:n ]
        data = (; y, x, species)

        fit = drm(bf(@formula(y ~ x + (1 | p | species)),
                     @formula(sigma ~ 1 + (1 | p | species))),
                  DRModels.Gamma(); data = data, se = false)

        # The sigma slot IS the shape here (≈ true α ≈ 4.95), not σ ≈ 1/√α ≈ 0.45.
        @test 3.5 < fit.scales[:sigma][1] < 6.5

        Random.seed!(1)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)     # was mean ≈ 1.26, std ≈ 0.09 before the fix
        @test ks_ok(r)          # was √n·D ≈ 46 before the fix (threshold 1.7)
    end

    @testset "Beta — continuous PIT (precision φ = σ⁻²)" begin
        Random.seed!(404)
        n = 2000
        x = randn(n)
        β = [0.3, 0.7]; φ = 15.0
        μ = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
        y = Float64.([rand(Distributions.Beta(μi * φ, (1 - μi) * φ)) for μi in μ])
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), DRModels.Beta(); data = data)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "NegBinomial2 — randomized discrete PIT (size θ = scales[:sigma])" begin
        Random.seed!(505)
        n = 3000
        x = randn(n)
        β = [0.6, 0.5]; θ = 4.0          # size; var = μ + μ²/θ
        μ = exp.(β[1] .+ β[2] .* x)
        y = Float64.([rand(Distributions.NegativeBinomial(θ, θ / (θ + μi))) for μi in μ])
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = data)
        Random.seed!(11)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "TruncatedNegBinomial2 — randomized discrete PIT (y ≥ 1)" begin
        Random.seed!(606)
        n = 3000
        x = randn(n)
        β = [0.7, 0.4]; θ = 5.0
        μ = exp.(β[1] .+ β[2] .* x)
        # draw from zero-truncated NB2 by rejection
        y = Vector{Float64}(undef, n)
        for i in 1:n
            d = Distributions.NegativeBinomial(θ, θ / (θ + μ[i]))
            yi = 0
            while yi == 0
                yi = rand(d)
            end
            y[i] = yi
        end
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), TruncatedNegBinomial2(); data = data)
        Random.seed!(12)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "Binomial — randomized discrete PIT" begin
        Random.seed!(707)
        n = 3000
        x = randn(n)
        β = [-0.2, 0.8]; ntrials = 12
        p = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
        succ = Float64.([rand(Distributions.Binomial(ntrials, pi)) for pi in p])
        fail = ntrials .- succ
        data = (; succ, fail, x)

        fit = drm(bf(@formula(cbind(succ, fail) ~ x)), DRModels.Binomial(); data = data)
        Random.seed!(13)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "BetaBinomial — randomized discrete PIT (φ = σ⁻²)" begin
        Random.seed!(808)
        n = 3000
        x = randn(n)
        β = [0.1, 0.6]; φ = 12.0; ntrials = 15
        p = 1 ./ (1 .+ exp.(-(β[1] .+ β[2] .* x)))
        succ = Float64.([rand(Distributions.BetaBinomial(ntrials, pi * φ, (1 - pi) * φ)) for pi in p])
        fail = ntrials .- succ
        data = (; succ, fail, x)

        fit = drm(bf(@formula(cbind(succ, fail) ~ x), @formula(sigma ~ 1)),
                  DRModels.BetaBinomial(); data = data)
        Random.seed!(14)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "CumulativeLogit — ordinal randomized PIT" begin
        Random.seed!(909)
        n = 4000
        x = randn(n)
        slope = 0.9
        cuts = [-1.2, 0.0, 1.3]          # K = 4 categories
        K = length(cuts) + 1
        η = slope .* x
        logistic(z) = 1 / (1 + exp(-z))
        y = Vector{Float64}(undef, n)
        for i in 1:n
            u = rand(); acc = 0.0; cat = K
            for k in 1:K
                pk = k == 1 ? logistic(cuts[1] - η[i]) :
                     k == K ? 1 - logistic(cuts[end] - η[i]) :
                     logistic(cuts[k] - η[i]) - logistic(cuts[k-1] - η[i])
                acc += pk
                if u <= acc
                    cat = k
                    break
                end
            end
            y[i] = cat
        end
        data = (; y, x)

        fit = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = data)
        Random.seed!(15)
        r = residuals(fit; type = :quantile)
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "ZeroOneBeta — atomic randomized PIT" begin
        Random.seed!(1010)
        n = 4000
        μb = 0.55; φ = 12.0; zoi = 0.25; coi = 0.4   # constant params
        y = Vector{Float64}(undef, n)
        for i in 1:n
            if rand() < zoi
                y[i] = rand() < coi ? 1.0 : 0.0
            else
                y[i] = rand(Distributions.Beta(μb * φ, (1 - μb) * φ))
            end
        end
        data = (; y)

        fit = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1), @formula(zoi ~ 1),
                     @formula(coi ~ 1)), ZeroOneBeta(); data = data)
        Random.seed!(16)
        r = residuals(fit; type = :quantile)
        @test length(r) == n
        @test all(isfinite, r)
        @test moments_ok(r)
        @test ks_ok(r)
    end

    @testset "discrete randomization is rng-reproducible" begin
        Random.seed!(2020)
        n = 500
        x = randn(n)
        λ = exp.(0.3 .+ 0.4 .* x)
        y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
        fit = drm(bf(@formula(y ~ x)), DRModels.Poisson(); data = (; y, x))
        r1 = residuals(fit; type = :quantile, rng = MersenneTwister(99))
        r2 = residuals(fit; type = :quantile, rng = MersenneTwister(99))
        @test r1 == r2
    end

    @testset "Tweedie scoped out — clear error" begin
        Random.seed!(1111)
        n = 400
        x = randn(n)
        μ = exp.(0.5 .+ 0.3 .* x)
        # Tweedie compound Poisson–Gamma draw (p = 1.5)
        p = 1.5; ϕ = 1.0
        y = Vector{Float64}(undef, n)
        for i in 1:n
            λ = μ[i]^(2 - p) / (ϕ * (2 - p))
            γ = ϕ * (p - 1) * μ[i]^(p - 1)
            sh = (2 - p) / (p - 1)
            N = rand(Distributions.Poisson(λ))
            y[i] = N == 0 ? 0.0 : rand(Distributions.Gamma(N * sh, γ))
        end
        data = (; y, x)
        tfit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Tweedie(); data = data)
        @test_throws ArgumentError residuals(tfit; type = :quantile)
    end

    @testset "unknown type throws" begin
        Random.seed!(7)
        n = 200
        x = randn(n)
        y = 0.5 .- 0.8 .* x .+ randn(n)
        data = (; y, x)
        fit = drm(bf(@formula(y ~ x)), Gaussian(); data = data)
        @test_throws ArgumentError residuals(fit; type = :bogus)
    end
end
