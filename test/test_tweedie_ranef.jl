# test_tweedie_ranef.jl — iid random effects on the Tweedie mean (#563 S8).
#
# drmTMB 0.7.0 ground truth (tests/testthat/test-tweedie-location-scale.R,
# `validate_tweedie_mu_random_terms()` in R/drmTMB.R): tweedie() `mu` supports
# an ordinary random intercept `(1 | id)` and (separately) an INDEPENDENT
# random slope `(0 + x | id)`, but NOT the correlated form `(1 + x | id)` —
# that is explicitly rejected ("Only independent tweedie() mu random
# intercepts and slopes are implemented in this slice."). `sigma`/`nu`
# formulas, and structured (phylo/relmat/animal/spatial) markers on `mu`,
# remain rejected outright ("tweedie random effects are not implemented" /
# no structured route wired). This test matches that same-target scope:
# `(1 | g)` is implemented and checked against the R fixture below;
# `(1 + x | g)`, RE on sigma/nu, and structured markers are refusal tests.
#
# R oracle fixtures: test/parity/fixtures/tweedie-mu-ranef/ (intercept) and
# test/parity/fixtures/tweedie-mu-slope-ranef/ (independent slope), each
# generated and fit by that directory's own gen_data.R —
#
#   Rscript test/parity/fixtures/tweedie-mu-ranef/gen_data.R
#   Rscript test/parity/fixtures/tweedie-mu-slope-ranef/gen_data.R
#
# with drmTMB 0.7.0 (`packageVersion("drmTMB")`), giving:
#
# intercept `(1 | id)` cell:
#   coef(fit, "mu")    = (0.5010962, 0.4022788)
#   coef(fit, "sigma") = -0.001603915          # log sigma; phi = exp(2*.) = 0.99680
#   coef(fit, "nu")    = -0.02398887           # logit-(1,2); p = 1.49400
#   sdpars$mu[["(1 | id)"]] = 0.4369495
#   logLik(fit)         = -572.2594
#   opt$convergence = 0, sdr$pdHess = TRUE
#
# independent slope `(0 + x | id)` cell:
#   coef(fit, "mu")    = (0.5411615, 0.3255704)
#   coef(fit, "sigma") = 0.03020911            # log sigma; phi = exp(2*.) = 1.06136
#   coef(fit, "nu")    = -0.1181794            # logit-(1,2); p = 1.47050
#   sdpars$mu[["(0 + x | id)"]] = 0.7054266
#   logLik(fit)         = -591.8245
#   opt$convergence = 0, sdr$pdHess = TRUE
#
# TOLERANCE: drmTMB's `(1 | id)` route is TMB's Laplace approximation (one
# saddlepoint per group, exact analytic Hessian); DRModels.jl's route below is
# 32-node non-adaptive Gauss–Hermite quadrature per group (the same scheme as
# the existing Poisson/Gamma `(1 | g)` routes in src/poisson.jl,
# src/gamma.jl). These are two DIFFERENT marginal-likelihood approximations
# of the same integral, not two runs of the same one — so exact bit parity is
# not the bar; same-target numerical agreement is. n_each = 8 obs/group keeps
# each group posterior close to Gaussian, so both approximations land close
# to the same optimum; the MEASURED gap (both engines converged) was:
#   intercept `(1 | id)` fixture:      |Δβ| <= 1.4e-4, |Δ log σ| = 1.7e-6,
#     |Δ logit p| = 3.3e-5, |Δ sd_id| = 8.9e-4, |Δ loglik| = 0.052.
#   independent slope `(0 + x | id)` fixture: |Δβ| <= 2.1e-4,
#     |Δ log σ| = 6.6e-5, |Δ logit p| = 3.3e-4, |Δ sd_slope| = 2.8e-3
#     (0.39% relative), |Δ loglik| = 0.076.
# Tolerances below give that margin roughly 5-10x headroom, not zero slack.

module TestTweedieRanef

using DRModels
using Test
using DelimitedFiles: readdlm

const FIXTURE = joinpath(@__DIR__, "parity", "fixtures", "tweedie-mu-ranef")
const FIXTURE_SLOPE = joinpath(@__DIR__, "parity", "fixtures", "tweedie-mu-slope-ranef")

function _load(dir)
    raw, header = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    j = Dict(c => i for (i, c) in enumerate(cols))
    id = string.(raw[:, j[:id]])
    x = Float64[parse(Float64, string(v)) for v in raw[:, j[:x]]]
    y = Float64[parse(Float64, string(v)) for v in raw[:, j[:y]]]
    return (; id, x, y)
end

@testset "Tweedie mu random effects (#563 S8)" begin
    data = _load(FIXTURE)

    @testset "(1 | g) random intercept — same-target vs drmTMB 0.7.0" begin
        fit = drm(bf(@formula(y ~ x + (1 | id)), @formula(nu ~ 1)), Tweedie(); data = data)

        @test isfinite(loglik(fit))

        # R oracle (see fixture header above).
        β_R = [0.5010962, 0.4022788]
        logσ_R = -0.001603915
        logitp_R = -0.02398887
        sdid_R = 0.4369495
        loglik_R = -572.2594

        β̂ = coef(fit, :mu)
        @test β̂[1] ≈ β_R[1] atol = 1e-3
        @test β̂[2] ≈ β_R[2] atol = 1e-3
        @test coef(fit, :sigma)[1] ≈ logσ_R atol = 1e-3
        @test coef(fit, :nu)[1] ≈ logitp_R atol = 1e-3
        @test exp(coef(fit, :resd)[1]) ≈ sdid_R rtol = 0.01
        @test loglik(fit) ≈ loglik_R atol = 0.3
    end

    @testset "(1 + x | g) correlated slope — refused, matches drmTMB" begin
        @test_throws Exception drm(
            bf(@formula(y ~ x + (1 + x | id)), @formula(nu ~ 1)), Tweedie(); data = data)
    end

    @testset "(0 + x | g) independent random slope — same-target vs drmTMB 0.7.0" begin
        data_slope = _load(FIXTURE_SLOPE)
        fit = drm(bf(@formula(y ~ x + (0 + x | id)), @formula(nu ~ 1)), Tweedie(); data = data_slope)

        @test isfinite(loglik(fit))

        # R oracle (see fixture header above).
        β_R = [0.5411615, 0.3255704]
        logσ_R = 0.03020911
        logitp_R = -0.1181794
        sdslope_R = 0.7054266
        loglik_R = -591.8245

        β̂ = coef(fit, :mu)
        @test β̂[1] ≈ β_R[1] atol = 1e-3
        @test β̂[2] ≈ β_R[2] atol = 1e-3
        @test coef(fit, :sigma)[1] ≈ logσ_R atol = 1e-3
        @test coef(fit, :nu)[1] ≈ logitp_R atol = 1e-3
        @test exp(coef(fit, :resd)[1]) ≈ sdslope_R rtol = 0.01
        @test loglik(fit) ≈ loglik_R atol = 0.3
    end

    @testset "random effect on sigma — refused" begin
        @test_throws Exception drm(
            bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | id)), @formula(nu ~ 1)),
            Tweedie(); data = data)
    end

    @testset "random effect on nu (power) — refused" begin
        @test_throws Exception drm(
            bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1 + (1 | id))),
            Tweedie(); data = data)
    end

    @testset "structured (phylo/relmat/…) marker on mu — refused" begin
        @test_throws Exception drm(
            bf(@formula(y ~ x + relmat(1 | id)), @formula(nu ~ 1)), Tweedie(); data = data)
    end
end

end # module
