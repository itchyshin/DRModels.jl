# Issue #706 (test part): classic lme4 teaching-dataset twins, cbpp and sleepstudy.
# drmTMB ships the same two comparisons (tests/testthat/test-comparators-phase19.R);
# DRModels had none. Datasets are vendored under test/fixtures/lme4/ (see README there).
#
# Reference values below are literals from lme4 2.0.1 (glmer / lmer) and drmTMB 0.7.1
# (`devtools`-free: installed package), each named with its integrator. Regenerate with
#   lme4::glmer(cbind(incidence, size - incidence) ~ period + (1 | herd), lme4::cbpp,
#               family = binomial, nAGQ = k)      # k = 1, 3, 5, 25
#   lme4::lmer(Reaction ~ Days + (Days | Subject), lme4::sleepstudy, REML = FALSE / TRUE)
#
# Integrator matching (cbpp):
#   DRModels default Binomial `(1 | g)` route = per-group adaptive GHQ with K = 5
#     (`_BINOMIAL_RANEF_AGHQ_K`, raised from 3 in #908)  <->  glmer nAGQ = 5
#   DRModels internal K = 3 (`_fit_binomial_ranef(...; nq = 3)`, the pre-#908 default)  <->  glmer nAGQ = 3
#   DRModels `marginal = :Laplace` / internal K = 1        <->  glmer nAGQ = 1, drmTMB (Laplace)
#   DRModels internal K = 25 (`_fit_binomial_ranef(...; nq = 25)`)  <->  glmer nAGQ = 25
# lme4's logLik for nAGQ > 1 drops the saturated-binomial term
#   sum(dbinom(y, n, y/n, log = TRUE)) = -41.97835377
# (nAGQ = 1 logLik does not: -92.0266 matches drmTMB/DRModels Laplace as is). We add it back
# to the lme4 nAGQ > 1 logLik; the corrected nAGQ = 25 value (-91.98336904) was confirmed
# against an independent `integrate()` marginal likelihood at lme4's own estimates
# (-91.98336904, R, rel.tol 1e-12).
using DRModels
using Test

function _read_csv_lme4(name)
    lines = readlines(joinpath(@__DIR__, "fixtures", "lme4", name))
    hdr = [strip(c, '"') for c in split(lines[1], ',')]
    cols = [String[strip(split(l, ',')[j], '"') for l in lines[2:end]] for j in eachindex(hdr)]
    return Dict(Symbol(h) => c for (h, c) in zip(hdr, cols))
end

@testset "lme4 twins (#706)" begin
    cb = _read_csv_lme4("cbpp.csv")
    incidence = parse.(Float64, cb[:incidence]); size_ = parse.(Float64, cb[:size])
    cbpp = (; herd = cb[:herd], period = cb[:period], incidence,
              fail = size_ .- incidence)

    @testset "cbpp: binomial (1 | herd) vs glmer" begin
        f0 = @formula(cbind(incidence, fail) ~ period + (1 | herd))
        X = hcat(ones(56), [cbpp.period[i] == p ? 1.0 : 0.0 for i in 1:56, p in ("2", "3", "4")])
        g = parse.(Int, cbpp.herd)
        sat = -41.97835377            # saturated binomial logLik (see header)

        # ---- references (literals) ------------------------------------------------
        # glmer nAGQ = 1 (Laplace): logLik as reported, drmTMB 0.7.1 (Laplace) alongside
        ref1 = (ll = -92.02656639, b = [-1.3983428642, -0.9919249752, -1.1282162161, -1.5797454139], sd = 0.6420699266)
        drmtmb1 = (ll = -92.02628186, b = [-1.3985321423, -0.9923327349, -1.1286720840, -1.5803138852], sd = 0.6422614484)
        # glmer nAGQ = 3 (AGHQ): logLik as reported is offset by `sat`
        ref3 = (ll = -50.03931072 + sat, b = [-1.3980373189, -0.9923740818, -1.1287976432, -1.5806352863], sd = 0.6439702368)
        # glmer nAGQ = 5 (AGHQ; lme4 2.0.1, same `sat` offset added back: logLik + sat = -91.98403769)
        ref5 = (ll = -91.98403769, b = [-1.399201882, -0.991438021, -1.127859471, -1.579506387], sd = 0.6473692007)
        # glmer nAGQ = 25 (AGHQ)
        ref25 = (ll = -50.00501527 + sat, b = [-1.399223728, -0.991408884, -1.127809594, -1.579480951], sd = 0.6475199134)

        # ---- K = 1: Laplace. glmer (PIRLS Laplace) and TMB differ at ~1e-3 (drmTMB's own
        # comparator test uses tol 1e-3 for the same reason), so DRModels is tied tightly
        # to drmTMB (same TMB-convention Laplace) and loosely to glmer.
        fl = drm(bf(f0), Binomial(); data = cbpp, marginal = :Laplace)
        @test loglik(fl) ≈ drmtmb1.ll atol = 1e-4
        @test coef(fl, :mu) ≈ drmtmb1.b atol = 1e-4
        @test re_sd(fl)[:herd] ≈ drmtmb1.sd atol = 1e-3
        @test loglik(fl) ≈ ref1.ll atol = 1e-3           # glmer: PIRLS vs TMB Laplace, ~3e-4
        @test coef(fl, :mu) ≈ ref1.b atol = 1e-3         # measured max 6e-4
        @test re_sd(fl)[:herd] ≈ ref1.sd atol = 1e-3     # measured 2e-4

        # ---- K = 5: DRModels default route (#908) vs glmer nAGQ = 5. Both are adaptive GHQ with
        # 5 nodes but glmer optimises fixed effects inside PIRLS, so the fixed effects
        # agree to ~1e-5 rather than exactly.
        f5 = drm(bf(f0), Binomial(); data = cbpp)
        @test DRModels._BINOMIAL_RANEF_AGHQ_K == 5
        @test loglik(f5) ≈ ref5.ll atol = 1e-4
        @test coef(f5, :mu) ≈ ref5.b atol = 1e-4
        @test re_sd(f5)[:herd] ≈ ref5.sd atol = 1e-3

        # ---- K = 3: the pre-#908 default, still reachable through the internal fitter.
        f3 = DRModels._fit_binomial_ranef(DRModels.Binomial(), incidence, size_, X, g, 15,
                                          ["(Intercept)", "period2", "period3", "period4"], :herd, 1e-8; nq = 3)
        @test loglik(f3) ≈ ref3.ll atol = 1e-4
        @test coef(f3, :mu) ≈ ref3.b atol = 1e-4
        @test re_sd(f3)[:herd] ≈ ref3.sd atol = 1e-3

        # ---- K = 25: internal fitter (no public K knob for Binomial) vs glmer nAGQ = 25.
        f25 = DRModels._fit_binomial_ranef(DRModels.Binomial(), incidence, size_, X, g, 15,
                                           ["(Intercept)", "period2", "period3", "period4"], :herd, 1e-8; nq = 25)
        @test loglik(f25) ≈ ref25.ll atol = 1e-4
        @test f25.theta[1:4] ≈ ref25.b atol = 1e-4        # measured max 7e-6
        @test exp(f25.theta[5]) ≈ ref25.sd atol = 1e-3    # measured 1.5e-6
        # the K = 5 default sits within 0.002 nat of the K = 25 (near-exact) logLik (measured 7e-4;
        # the old K = 3 default was ~0.03 nat off)
        @test abs(loglik(f5) - loglik(f25)) < 0.002
        @test abs(loglik(f5) - loglik(f25)) < abs(loglik(f3) - loglik(f25))
    end

    @testset "sleepstudy: Gaussian (1 + Days | Subject) vs lmer" begin
        ss = _read_csv_lme4("sleepstudy.csv")
        sleep = (; Reaction = parse.(Float64, ss[:Reaction]), Days = parse.(Float64, ss[:Days]),
                   Subject = ss[:Subject])
        # Explicit-intercept spelling; the lme4 spelling `(Days | Subject)` is checked below.
        f = @formula(Reaction ~ Days + (1 + Days | Subject))

        # lmer REML = FALSE (ML, exact profiled Cholesky) and drmTMB 0.7.1 (Laplace of a
        # Gaussian = exact); lmer sigma 25.59190704, drmTMB sigma 25.59181563.
        lm_ml = (ll = -875.9696722, b = [251.40510485, 10.46728596],
                 sd_int = 23.77975960, sd_slope = 5.71679851, rho = 0.08132109, sigma = 25.59190704)
        tmb_ml = (ll = -875.9696722, b = [251.40510485, 10.46728596],
                  sd_int = 23.780566999, sd_slope = 5.716834542, rho = 0.08132007536, sigma = 25.59181563)
        fit = drm(bf(f), Gaussian(); data = sleep)
        @test loglik(fit) ≈ lm_ml.ll atol = 1e-4
        @test coef(fit, :mu) ≈ lm_ml.b atol = 1e-4
        # SD tolerance: measured 8.1e-4 on the intercept SD (within the stated 1e-3;
        # the ML surface is flat in the RE variance, lmer stops at rel. 1e-6 in theta).
        sds = re_sd(fit)
        @test sds[:Subject_intercept] ≈ lm_ml.sd_int atol = 1e-3
        @test sds[:Subject_slope] ≈ lm_ml.sd_slope atol = 1e-3
        @test sds[:Subject_intercept] ≈ tmb_ml.sd_int atol = 1e-4   # drmTMB agrees to 1e-9
        @test sds[:Subject_slope] ≈ tmb_ml.sd_slope atol = 1e-4
        Σ = vc(fit)[:Subject]
        @test Σ[1, 2] / sqrt(Σ[1, 1] * Σ[2, 2]) ≈ lm_ml.rho atol = 1e-3
        @test exp(coef(fit, :sigma)[1]) ≈ lm_ml.sigma atol = 1e-3   # measured 9e-5

        # With the implicit intercept (#890), lme4's own spelling `(Days | Subject)` is the
        # same model as `(1 + Days | Subject)`.
        fit_imp = drm(bf(@formula(Reaction ~ Days + (Days | Subject))), Gaussian(); data = sleep)
        @test loglik(fit_imp) ≈ loglik(fit) atol = 1e-6
        @test coef(fit_imp, :mu) ≈ coef(fit, :mu) atol = 1e-6

        # REML (correlated random-slope Gaussian route). References are literals:
        # lmer REML: logLik -871.814136, SD 24.74065800 / 5.92213765, rho 0.06555124, sigma 25.59179572;
        # drmTMB REML: logLik -871.814136, SD 24.740451464 / 5.922133104, rho 0.06555132, sigma 25.59181563.
        lm_reml = (ll = -871.8141, b = [251.40510485, 10.46728596], sd_int = 24.74065800,
                   sd_slope = 5.92213765, rho = 0.06555124, sigma = 25.59179572)
        fr = drm(bf(f), Gaussian(); data = sleep, method = :REML)
        @test loglik(fr) ≈ lm_reml.ll atol = 1e-4
        @test coef(fr, :mu) ≈ lm_reml.b atol = 1e-4
        sdr = re_sd(fr)
        @test sdr[:Subject_intercept] ≈ lm_reml.sd_int atol = 1e-3
        @test sdr[:Subject_slope] ≈ lm_reml.sd_slope atol = 1e-3
        Σr = vc(fr)[:Subject]
        @test Σr[1, 2] / sqrt(Σr[1, 1] * Σr[2, 2]) ≈ lm_reml.rho atol = 1e-3
        @test exp(coef(fr, :sigma)[1]) ≈ lm_reml.sigma atol = 1e-3
        # The ML path is untouched by the REML term.
        @test loglik(drm(bf(f), Gaussian(); data = sleep)) ≈ lm_ml.ll atol = 1e-4

        # Gaussian random-INTERCEPT REML must be unchanged (pre-change literals, 1e-10).
        fri = drm(bf(@formula(Reaction ~ Days + (1 | Subject))), Gaussian(); data = sleep, method = :REML)
        @test loglik(fri) ≈ -893.2325426974637 atol = 1e-10
        @test coef(fri, :mu) ≈ [251.40510484848465, 10.467285959595994] atol = 1e-10
        @test coef(fri, :sigma)[1] ≈ 3.4337043872235142 atol = 1e-10
        @test re_sd(fri)[:Subject] ≈ 37.12382669423763 atol = 1e-10
    end
end
