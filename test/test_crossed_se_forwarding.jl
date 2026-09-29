# Regression for the Gamma/Beta/BetaBinomial siblings of issue #761: each of
# `_fit_gamma_crossed_laplace`, `_fit_beta_crossed_laplace`, and
# `_fit_betabinomial_crossed_laplace` (src/sparse_laplace_glmm.jl) defaulted
# `se::Bool = false`, and the corresponding family dispatcher (src/gamma.jl,
# src/beta.jl, src/betabinomial.jl) never forwarded the caller's `se` argument
# down to it -- so a crossed `(1 | g) + (1 | h)` fit always skipped the
# Hessian/vcov step regardless of what `drm(...; se = ...)` was asked for,
# reporting silent all-`Inf` standard errors with no warning (same class as
# #761's Binomial case; only Poisson and NB2 dispatchers forwarded `se`
# correctly).
#
# Fix: flip each crossed fitter's own default to `se = true` (matching
# Poisson/NB2), and have the three family dispatchers forward `se = se` at
# their crossed-dispatch call site, so `se = false` still works when asked.
using DRModels
using Test
using Random
using LinearAlgebra
using Logging
import Distributions

logistic(x) = 1 / (1 + exp(-x))

@testset "crossed random-effect SEs: Gamma, Beta, BetaBinomial forward `se` (twin of #761)" begin

    @testset "Gamma: well-identified crossed fit" begin
        rng = MersenneTwister(761_01)
        G, H, n = 28, 24, 2400
        x = randn(rng, n)
        gids = rand(rng, 1:G, n); hids = rand(rng, 1:H, n)
        g = [Symbol("g", j) for j in gids]; h = [Symbol("h", j) for j in hids]
        β = [0.2, 0.35]; σg = 0.30; σh = 0.25
        bg = σg .* randn(rng, G); bh = σh .* randn(rng, H)
        η = [β[1] + β[2] * x[i] + bg[gids[i]] + bh[hids[i]] for i in 1:n]
        μ = exp.(η)
        shape = 7.0
        y = Float64.([rand(rng, Distributions.Gamma(shape, μ[i] / shape)) for i in 1:n])
        data = (; y, x, g, h)
        form = bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1))

        # Default (se not passed, i.e. drm()'s own default se = true): must
        # now reach the Hessian/vcov step and return finite SEs.
        fit = drm(form, Gamma(); data = data)
        @test fit.converged
        se = stderror(fit)
        @test all(isfinite, se)
        @test all(>(0), se)

        # se = false must still be honoured (all-Inf, no crash).
        fit_nose = drm(form, Gamma(); data = data, se = false)
        @test fit_nose.converged
        @test all(isinf, stderror(fit_nose))
    end

    @testset "Gamma: degenerate second grouping warns instead of silently failing" begin
        rng = MersenneTwister(761_02)
        G, n = 28, 2400
        x = randn(rng, n)
        gids = rand(rng, 1:G, n)
        g = [Symbol("g", j) for j in gids]
        h = fill(:h1, n)                      # single level: aliased with the intercept
        β = [0.2, 0.35]; σg = 0.30
        bg = σg .* randn(rng, G)
        η = [β[1] + β[2] * x[i] + bg[gids[i]] for i in 1:n]
        μ = exp.(η)
        shape = 7.0
        y = Float64.([rand(rng, Distributions.Gamma(shape, μ[i] / shape)) for i in 1:n])
        data = (; y, x, g, h)
        form = bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1))

        fit = @test_logs (:warn, r"(numerically singular|not positive definite|not trustworthy)"i) match_mode = :any drm(
            form, Gamma(); data = data)
        # The identified fixed effects stay finite; the aliased random-intercept
        # coordinate is flagged by the warning above rather than failing
        # silently (the guard's pseudo-inverse may itself return a finite, but
        # explicitly-warned-as-untrustworthy, value for that coordinate).
        se = stderror(fit)
        @test all(isfinite, se[1:2])
    end

    @testset "Beta: well-identified crossed fit" begin
        rng = MersenneTwister(761_03)
        G, H, n = 28, 24, 2400
        x = randn(rng, n)
        gids = rand(rng, 1:G, n); hids = rand(rng, 1:H, n)
        g = [Symbol("g", j) for j in gids]; h = [Symbol("h", j) for j in hids]
        β = [0.1, 0.4]; σg = 0.30; σh = 0.25
        bg = σg .* randn(rng, G); bh = σh .* randn(rng, H)
        η = [β[1] + β[2] * x[i] + bg[gids[i]] + bh[hids[i]] for i in 1:n]
        p = logistic.(η)
        φ = 25.0
        y = Float64.([rand(rng, Distributions.Beta(p[i] * φ, (1 - p[i]) * φ)) for i in 1:n])
        data = (; y, x, g, h)
        form = bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1))

        fit = drm(form, Beta(); data = data)
        @test fit.converged
        se = stderror(fit)
        @test all(isfinite, se)
        @test all(>(0), se)

        fit_nose = drm(form, Beta(); data = data, se = false)
        @test fit_nose.converged
        @test all(isinf, stderror(fit_nose))
    end

    @testset "Beta: degenerate second grouping warns instead of silently failing" begin
        rng = MersenneTwister(761_04)
        G, n = 28, 2400
        x = randn(rng, n)
        gids = rand(rng, 1:G, n)
        g = [Symbol("g", j) for j in gids]
        h = fill(:h1, n)
        β = [0.1, 0.4]; σg = 0.30
        bg = σg .* randn(rng, G)
        η = [β[1] + β[2] * x[i] + bg[gids[i]] for i in 1:n]
        p = logistic.(η)
        φ = 25.0
        y = Float64.([rand(rng, Distributions.Beta(p[i] * φ, (1 - p[i]) * φ)) for i in 1:n])
        data = (; y, x, g, h)
        form = bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1))

        fit = @test_logs (:warn, r"(numerically singular|not positive definite|not trustworthy)"i) match_mode = :any drm(
            form, Beta(); data = data)
        se = stderror(fit)
        @test all(isfinite, se[1:2])
        @test any(!isfinite, se)
    end

    @testset "BetaBinomial: well-identified crossed fit" begin
        rng = MersenneTwister(761_05)
        G, H, n = 28, 24, 2400
        x = randn(rng, n)
        gids = rand(rng, 1:G, n); hids = rand(rng, 1:H, n)
        g = [Symbol("g", j) for j in gids]; h = [Symbol("h", j) for j in hids]
        β = [0.10, 0.40]; σg = 0.35; σh = 0.25
        bg = σg .* randn(rng, G); bh = σh .* randn(rng, H)
        η = [β[1] + β[2] * x[i] + bg[gids[i]] + bh[hids[i]] for i in 1:n]
        μ = logistic.(η)
        precision = 18.0
        ntr = fill(8, n)
        successes = Float64.([rand(rng, Distributions.BetaBinomial(ntr[i], μ[i] * precision, (1 - μ[i]) * precision)) for i in 1:n])
        failures = Float64.(ntr) .- successes
        data = (; successes, failures, x, g, h)
        form = bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1))

        fit = drm(form, BetaBinomial(); data = data)
        @test fit.converged
        se = stderror(fit)
        @test all(isfinite, se)
        @test all(>(0), se)

        fit_nose = drm(form, BetaBinomial(); data = data, se = false)
        @test fit_nose.converged
        @test all(isinf, stderror(fit_nose))
    end

    @testset "BetaBinomial: degenerate second grouping warns instead of silently failing" begin
        rng = MersenneTwister(761_06)
        G, n = 28, 2400
        x = randn(rng, n)
        gids = rand(rng, 1:G, n)
        g = [Symbol("g", j) for j in gids]
        h = fill(:h1, n)
        β = [0.10, 0.40]; σg = 0.35
        bg = σg .* randn(rng, G)
        η = [β[1] + β[2] * x[i] + bg[gids[i]] for i in 1:n]
        μ = logistic.(η)
        precision = 18.0
        ntr = fill(8, n)
        successes = Float64.([rand(rng, Distributions.BetaBinomial(ntr[i], μ[i] * precision, (1 - μ[i]) * precision)) for i in 1:n])
        failures = Float64.(ntr) .- successes
        data = (; successes, failures, x, g, h)
        form = bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1))

        fit = @test_logs (:warn, r"(numerically singular|not positive definite|not trustworthy)"i) match_mode = :any drm(
            form, BetaBinomial(); data = data)
        se = stderror(fit)
        @test all(isfinite, se[1:2])
        @test any(!isfinite, se)
    end
end
