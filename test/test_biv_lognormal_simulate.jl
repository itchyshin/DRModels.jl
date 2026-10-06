# #766-class: `simulate(fit)` / the parametric bootstrap threw `KeyError: key
# :mu not found` for a bivariate lognormal fit (`biv_lognormal()`), the same
# bug PR #840 found and fixed for bivariate Student-t (`biv_student()`).
#
# Root cause: the generic `_simulate_once` in gaussian_core.jl special-cases
# bivariate GAUSSIAN fits (`fam isa Gaussian && haskey(fit.scales, :sigma1)`)
# before falling through to `fit.means[:mu]` / `fit.scales[:sigma]` for every
# other family. A bivariate lognormal fit has `fit.family isa LogNormal` --
# never `Gaussian` -- even though `drm(f, LogNormal(); data)` (this family's
# whole fit) is built by delegating entirely to the bivariate Gaussian route
# on `log(y)` (`_lognormal_jacobian_shift`, src/bivariate_lognormal.jl), so its
# `means`/`scales` carry the SAME `:mu1`/`:mu2`/`:sigma1`/`:sigma2`/`:rho12`
# keys as bivariate Gaussian, never `:mu`/`:sigma`. That fallthrough threw
# `KeyError: key :mu not found` on the very first `simulate(fit)` call, and
# therefore on every parametric-bootstrap replicate too (`bootstrap_result`/
# `bootstrap_ci` draw via `simulate(fit0; rng)` before any refit).
#
# Fix: `src/bivariate_lognormal.jl` adds `_simulate_once(fit::DrmFit{LogNormal},
# rng; ...)`, more specific than the generic `fit::DrmFit` method, dispatching
# to a bivariate branch (draw log(Y) exactly as bivariate Gaussian, then
# exponentiate) when `fit.scales` has `:sigma1`, and to the untouched
# univariate lognormal draw otherwise.

using DRModels
using Test
using Random
using Statistics

@testset "biv_lognormal simulate()/bootstrap no longer throw KeyError" begin

    rng = MersenneTwister(5501)
    n = 4000
    x = randn(rng, n)
    mu1, mu2 = 0.4, 0.1
    b1, b2 = 0.3, -0.2
    s1, s2, rho = 0.35, 0.5, 0.4
    z1 = randn(rng, n)
    z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(rng, n)
    l1 = mu1 .+ b1 .* x .+ s1 .* z1   # log(y1)
    l2 = mu2 .+ b2 .* x .+ s2 .* z2   # log(y2)
    y1 = exp.(l1)
    y2 = exp.(l2)
    data = (; y1 = y1, y2 = y2, x = x)
    f = bf(; mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
           rho12 = @formula(rho12 ~ 1))

    fit = drm(f, LogNormal(); data = data)
    @test fit.converged

    @testset "simulate(fit) no longer throws KeyError (the #840-class gate)" begin
        ysim = simulate(fit; rng = MersenneTwister(1))
        @test ysim isa AbstractDict
        @test haskey(ysim, :mu1) && haskey(ysim, :mu2)
        @test length(ysim[:mu1]) == n
        @test all(isfinite, ysim[:mu1]) && all(isfinite, ysim[:mu2])
        # Both margins must stay strictly positive -- it's a lognormal draw.
        @test all(>(0), ysim[:mu1]) && all(>(0), ysim[:mu2])
    end

    @testset "simulated draws recover the marginal means/correlation at large n" begin
        # Draw many replicates and check the RESPONSE-SCALE (not log-scale)
        # marginal means and the LOG-SCALE residual correlation against the
        # data-generating values, within a loose Monte Carlo tolerance.
        nsim = 2000
        Y = simulate(fit; nsim = nsim, rng = MersenneTwister(2))
        @test Y isa AbstractVector
        @test length(Y) == nsim

        y1sim_mean = mean(mean(rep[:mu1]) for rep in Y)
        y2sim_mean = mean(mean(rep[:mu2]) for rep in Y)
        # E[Y] for lognormal(mu, sigma^2) is exp(mu + sigma^2/2); average over x
        # since mu1/mu2 here include a slope. Compare against the empirical
        # observed-data means instead of re-deriving the closed form, so the
        # test only pins the SHAPE of the draw (right family, right scale),
        # not a second copy of the lognormal moment formula.
        @test isapprox(y1sim_mean, mean(y1); rtol = 0.15)
        @test isapprox(y2sim_mean, mean(y2); rtol = 0.15)

        # Residual correlation, not the raw log(y) correlation: log(y1)/log(y2)
        # both carry the shared x-slope, so correlating them directly measures
        # a mix of that shared signal and the residual correlation. Subtract
        # each margin's own fitted (log-scale) mean first, exactly as rho12 is
        # defined (the log-residual correlation, per this file's own docstring).
        logcor_sim = mean(cor(log.(rep[:mu1]) .- fit.means[:mu1],
                               log.(rep[:mu2]) .- fit.means[:mu2]) for rep in Y)
        @test isapprox(logcor_sim, rho; atol = 0.08)
    end

    # `bootstrap_result`/`bootstrap_ci` report one row PER FITTED COEFFICIENT
    # (mu1: 2, mu2: 2, sigma1/sigma2/rho12: 1 each = 7 here), aggregated over
    # the B replicates -- not one row per replicate.
    ncoef = sum(length(last(p)) for p in fit.coefnames)

    @testset "bootstrap_result: replicates run and return finite endpoints" begin
        res = bootstrap_result(fit; data = data, B = 8, rng = MersenneTwister(3))
        @test res.failed == 0
        @test res.used == 8
        @test length(res.summary) == ncoef
        @test all(isfinite(r.lower) && isfinite(r.upper) for r in res.summary)
    end

    @testset "bootstrap_ci: same, via the CI-only surface" begin
        ci = bootstrap_ci(fit; data = data, B = 8, rng = MersenneTwister(4))
        @test length(ci) == ncoef
        @test all(isfinite(r.lower) && isfinite(r.upper) for r in ci)
    end

end
