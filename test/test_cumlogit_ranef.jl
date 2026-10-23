# test_cumlogit_ranef.jl — iid random effects on the cumulative-logit (ordinal)
# mean (#563 S8).
#
# drmTMB 0.7.0 ground truth (R/drmTMB.R `validate_cumulative_logit_mu_random_terms()`,
# tests/testthat/test-cumulative-logit.R, test-arc2a-mu-random-intercept.R,
# test-arc2b-mu-random-slope.R): `cumulative_logit()` `mu` supports an ordinary
# random intercept `(1 | id)` and (separately) an INDEPENDENT random slope
# `(0 + x | id)`, but NOT the correlated form `(1 + x | id)` — that is
# explicitly rejected ("Only independent `cumulative_logit()` `mu` random
# intercepts and slopes are implemented in this slice."). Random-effect scale
# formulae (`sd(id) ~ 1`) are rejected outright, matching drmTMB's own
# "Random-effect scale formulae are not implemented" guard. drmTMB *does* admit
# an unlabelled, intercept-only `phylo(1 | species)` structured effect on `mu`
# (`validate_ordinal_phylo_mu_structured_term()`) — that route IS now
# implemented (#563 S8 follow-on, test_cumlogit_phylo.jl) via the sparse
# augmented-state Laplace GLMM engine; the refusal test below covers
# `relmat`/`animal`/`spatial` structured markers only, which stay refused
# (DRModels.jl scope decision, not a drmTMB gap — drmTMB itself only admits `phylo`
# for `cumulative_logit()`).
#
# R oracle fixtures: test/parity/fixtures/cumlogit-mu-ranef/ (intercept) and
# test/parity/fixtures/cumlogit-mu-slope-ranef/ (independent slope), each
# generated and fit by that directory's own gen_data.R —
#
#   Rscript test/parity/fixtures/cumlogit-mu-ranef/gen_data.R
#   Rscript test/parity/fixtures/cumlogit-mu-slope-ranef/gen_data.R
#
# with drmTMB 0.7.0 (`packageVersion("drmTMB")`), giving:
#
# intercept `(1 | id)` cell (DGP copied from drmTMB's own Arc 2a sentinel,
# seed 9, n_id=45, n_each=18, K=4, cutpoints c(-1,0,1), sd_id=0.7):
#   coef(fit, "mu")      = 0.7988354
#   cutpoints             = -1.022415, 0.02801576, 1.012281
#   sdpars$mu[["(1 | id)"]] = 0.6876018
#   logLik(fit)           = -1026.092
#   opt$convergence = 0, sdr$pdHess = TRUE
#
# independent slope `(0 + x | id)` cell (DGP copied from drmTMB's own Arc 2b
# `base_slope(20260724)`, n_id=40, n_each=15, K=4, cutpoints c(-1,0,1),
# slope_sd=0.5):
#   coef(fit, "mu")      = 0.7990546
#   cutpoints             = -1.029216, 0.1064309, 1.104362
#   sdpars$mu[["(0 + x | id)"]] = 0.3335849
#   logLik(fit)           = -775.3768
#   opt$convergence = 0, sdr$pdHess = TRUE
#
# TOLERANCE: drmTMB's route is TMB's Laplace approximation (one saddlepoint
# per group, exact analytic Hessian); DRModels.jl's route below is 41-node
# per-group ADAPTIVE Gauss–Hermite quadrature (src/adaptive_ghq.jl; it was
# 32-node non-adaptive GHQ when these tolerances were set). These are two DIFFERENT marginal-likelihood
# approximations of the same integral, not two runs of the same one, so exact
# bit parity is not the bar; same-target numerical agreement is. n_each >= 15
# obs/group keeps each group posterior close to Gaussian, so both
# approximations should land close to the same optimum. Tolerances below
# mirror the tweedie ranef sibling slice's measured-gap-derived margins
# (atol=1e-2 on beta/cutpoints, rtol=0.05 on the RE-SD, atol=1.0 on loglik —
# looser than tweedie's because the ordinal likelihood's curvature is flatter
# per observation, so the two quadratures can disagree by a larger absolute
# amount while still targeting the same optimum).

module TestCumlogitRanef

using DRModels
using Test
using DelimitedFiles: readdlm

const FIXTURE = joinpath(@__DIR__, "parity", "fixtures", "cumlogit-mu-ranef")
const FIXTURE_SLOPE = joinpath(@__DIR__, "parity", "fixtures", "cumlogit-mu-slope-ranef")

function _load(dir)
    raw, header = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    j = Dict(c => i for (i, c) in enumerate(cols))
    id = string.(raw[:, j[:id]])
    x = Float64[parse(Float64, string(v)) for v in raw[:, j[:x]]]
    y = Float64[parse(Float64, string(v)) for v in raw[:, j[:y_int]]]
    return (; id, x, y)
end

@testset "CumulativeLogit mu random effects (#563 S8)" begin
    data = _load(FIXTURE)

    @testset "(1 | g) random intercept — same-target vs drmTMB 0.7.0" begin
        fit = drm(bf(@formula(y ~ x + (1 | id))), CumulativeLogit(); data = data)

        @test isfinite(loglik(fit))

        # R oracle (see fixture header above).
        β_R = 0.7988354
        cuts_R = [-1.022415, 0.02801576, 1.012281]
        sdid_R = 0.6876018
        loglik_R = -1026.092

        @test coef(fit, :mu)[1] ≈ β_R atol = 1e-2
        δ = coef(fit, :cutpoints)
        cutŝ = similar(δ); cutŝ[1] = δ[1]
        for k in 2:length(δ); cutŝ[k] = cutŝ[k-1] + exp(δ[k]); end
        @test cutŝ ≈ cuts_R atol = 1e-2
        @test exp(coef(fit, :resd)[1]) ≈ sdid_R rtol = 0.05
        @test loglik(fit) ≈ loglik_R atol = 1.0
    end

    @testset "(1 + x | g) correlated slope — refused, matches drmTMB" begin
        @test_throws Exception drm(
            bf(@formula(y ~ x + (1 + x | id))), CumulativeLogit(); data = data)
    end

    @testset "(0 + x | g) independent random slope — same-target vs drmTMB 0.7.0" begin
        data_slope = _load(FIXTURE_SLOPE)
        fit = drm(bf(@formula(y ~ x + (0 + x | id))), CumulativeLogit(); data = data_slope)

        @test isfinite(loglik(fit))

        # R oracle (see fixture header above).
        β_R = 0.7990546
        cuts_R = [-1.029216, 0.1064309, 1.104362]
        sdslope_R = 0.3335849
        loglik_R = -775.3768

        @test coef(fit, :mu)[1] ≈ β_R atol = 1e-2
        δ = coef(fit, :cutpoints)
        cutŝ = similar(δ); cutŝ[1] = δ[1]
        for k in 2:length(δ); cutŝ[k] = cutŝ[k-1] + exp(δ[k]); end
        @test cutŝ ≈ cuts_R atol = 1e-2
        @test exp(coef(fit, :resd)[1]) ≈ sdslope_R rtol = 0.05
        @test loglik(fit) ≈ loglik_R atol = 1.0
    end

    @testset "crossed/multiple random effects — refused" begin
        @test_throws Exception drm(
            bf(@formula(y ~ x + (1 | id) + (0 + x | id))), CumulativeLogit(); data = data)
    end

    @testset "structured phylo marker on mu — refused (scope decision, not a drmTMB gap)" begin
        @test_throws Exception drm(
            bf(@formula(y ~ x + relmat(1 | id))), CumulativeLogit(); data = data)
    end

    @testset "random-effect scale formula — refused" begin
        @test_throws Exception drm(
            bf(@formula(y ~ x), @formula(sd(id) ~ 1)), CumulativeLogit(); data = data)
    end
end

end # module
