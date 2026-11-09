# Non-Gaussian REML: what drmTMB 0.8.1 does, and whether Julia should match it

Date: 2026-11-09. Lane: R-Julia parity, research and decision. Base: wave-10 tip `0bfcde2bb`. Docs only.

## Recommendation

Keep refusing `method = :REML` on Binomial in DRModels.jl for now, and fix the refusal message (small, about one hour). Do not build a matching estimator yet. Three reasons. R's non-Gaussian REML is a joint-Laplace approximation that R itself labels `diagnostic_only` (no recovery or coverage evidence), and Julia's one non-Gaussian REML (Poisson) is a different approximation, so a Binomial match would put two estimators under one `:REML` label. The gain is 2 of 113 behind-cells. A clear refusal costs the user nothing they cannot get with `method = :ML`.

Revisit if drmTMB promotes binomial REML above `diagnostic_only`, or if a user asks for it. A design for that case is in section 5.

## 1. What `REML = TRUE` means in drmTMB 0.8.1 for non-Gaussian families

All line numbers are for drmTMB commit `67acc18a1` (the checkout is dirty, so I read `git show HEAD:`).

**Only binomial is admitted.** Every other non-Gaussian family aborts with "REML is implemented for univariate/bivariate Gaussian and binomial models" (`R/drmTMB.R:404-408`, and again in `drm_validate_reml_spec`, `R/drmTMB.R:3060-3065`). The parameter docs say the same: "Every other non-Gaussian family rejects `REML = TRUE`" (`R/drmTMB.R:185`).

**The binomial route is narrow.** Exactly one ordinary unlabelled `mu` random term: one intercept `(1 | g)` or one independent slope `(0 + x | g)`. Fixed-only, multiple-term, correlated or labelled blocks, structured effects and missing-data engines are refused (`R/drmTMB.R:3066-3080`, docs at `R/drmTMB.R:177-183`).

**The estimator is a joint Laplace, not exact REML.** `drm_apply_estimator_spec` puts `beta_mu` into the TMB `random` block next to the random effects (`R/drmTMB.R:1315-1348`). TMB then does one Laplace approximation over (u, beta) with a flat prior on beta. The code comment says this "IS the joint-Laplace restricted likelihood, which equals glmmTMB(REML = TRUE)" (`R/drmTMB.R:399-403`). Design doc 224 calls it O2 and states it is not the AGHQ plus Cox-Reid object, which is package-private and not what `REML = TRUE` runs (`docs/design/224-aghq-coxreid-nongaussian-reml-alignment.md:34-37`; `R/drmTMB.R:186-187`).

**Exactness is flagged in the fit object.** `drm_estimator_exact` returns `TRUE` only for Gaussian and bivariate Gaussian, and `FALSE` for everything else under REML (`R/drmTMB.R:1351-1360`). I confirmed `estimator_exact = FALSE` on binomial fits (section 3).

**Status is documented.** Doc 224 classes binomial REML `diagnostic_only`: the `glmmTMB(REML = TRUE)` match and finite-uncertainty checks "establish correct wiring, not recovery or coverage", and "No general non-Gaussian REML claim follows" (`docs/design/224-...md:256-259`). The roxygen says the route "has diagnostic parity and finite-uncertainty evidence, not calibrated interval or coverage evidence" (`R/drmTMB.R:183-185`). Doc 224 reports RE-SD agreement with `glmmTMB` of 7.3e-9 and a measured 42% cut in ML variance bias in one scoping fixture (`:229-233` and `:44-46`; I did not re-derive either figure).

**Labelling flaw to report to the R side.** `print.summary.drmTMB` shows "Laplace/Cox-Reid adjusted profile" for any non-exact REML (`R/methods.R:4494`). The binomial route is a joint-Laplace fold, which doc 224 says is not Cox-Reid. The label is wrong for the one route that uses it. Small fix, not made here.

## 2. The wave-10 cells and the Julia refusal

`julia_behind_ranked.tsv` at `origin/claude/gap-map-wave10-20261109` has exactly two non-Gaussian REML rows, both `R-only`, `REFUSES`, `ArgumentError`:

| cell | R logLik | R state |
|---|---|---|
| `binomial\|ordinary\|lab0\|q1\|sd-none\|cm0\|REML` | -261.3983 | pdHess TRUE, is_converged TRUE |
| `binomial\|slope\|lab0\|q1\|sd-none\|cm0\|REML` | -262.3518 | pdHess TRUE, is_converged TRUE |

Julia message: "drm (Binomial): unknown `method = :REML`. Binomial is ML-only; for the integrator use `marginal = :LA` ..., `:Laplace` ..., `:VA` (#136), or `:AGHQ` (#448)."

It is raised in `src/variational.jl:132-152` (`_reject_method_as_marginal`, the `:REML` branch at `:147-152`), called for Binomial at `src/binomial.jl:284`. The same call without `allow_reml` is used for Beta, Gamma and NB2 (`src/beta.jl:73`, `src/gamma.jl:77`, `src/negbinomial.jl:158`). Poisson and the shared ordinary-Laplace spine pass `allow_reml = true` (`src/poisson.jl:250`, `src/ordinary_laplace.jl:69`) and then refuse specific shapes with `_reject_reml_route` (`src/variational.jl:165-173`).

Two problems with the message. It calls `:REML` an "unknown method" and answers with a list of integrators, which does not tell the user how to get what they want (omit `method`, or `:ML`). It also contradicts drmTMB, which fits binomial REML. A Binomial test already pins the substring "ML-only" (`test/test_cox_reid_poisson_phylo.jl:207-218`), so keep that word.

Julia's only non-Gaussian REML is Poisson: the Cox-Reid adjusted profile, "maximises ℓ_ML − ½·log|I_ββ|" (`src/poisson.jl:89-96`), which Julia's own docs say over-corrects when clusters are plentiful (`src/poisson.jl:105-111`). R refuses Poisson REML. So the two engines already offer non-Gaussian REML on disjoint families with different approximations (doc 261 rows `phylo_count_large_p` poisson and `plain_binomial_nonphylo`, `docs/design/261-reml-by-route.md:122-125, 221-223`).

## 3. Check run (Totoro, 2026-11-09, about 27 s on one core, nice 10, thread caps 1; load was 61 of the 100 limit)

drmTMB 0.8.1 from `~/drm_campaign_20261108/Rlib`. 40 simulated data sets per cell (seeds fixed, script and per-fit output in `docs/dev-log/evidence/reml-nongauss-20261109/`). Each data set fitted with `REML = FALSE`, `REML = TRUE`, and `glmmTMB(REML = TRUE)`.

| cell | G x m | true SD | median SD, ML | median SD, REML | REML >= ML | `is_converged()` REML / ML | max abs gap to glmmTMB REML | `estimator_exact` |
|---|---|---|---|---|---|---|---|---|
| A binomial `(1 \| g)` | 15 x 8 | 1.0 | 1.053 | 1.156 | 40/40 | 1.00 / 1.00 | 1.4e-07 | FALSE |
| B binomial `(1 \| g)` | 40 x 6 | 0.6 | 0.658 | 0.693 | 39/39 | 0.97 / 0.97 | 3.8e-05 | FALSE |
| C binomial `(0 + x \| g)` | 20 x 10 | 0.8 | 0.819 | 0.898 | 37/37 | 0.95 / 0.93 | 5.3e-05 | FALSE |
| D gaussian `(1 \| g)`, control | 10 x 6 | 0.8 | 0.702 | 0.755 | 39/39 | 1.00 / 0.97 | 1.6e-06 | TRUE |
| E binomial `(1 \| g)` | 8 x 20 | 1.0 | 0.837 | 0.948 | 39/39 | 0.97 / 0.97 | 4.4e-05 | FALSE |

(Counts are fits with a usable ML SD above 1e-3 and complete output; a few fits dropped out per cell.)

What this shows, and only this:
- REML SD is at least the ML SD in every complete data set, as theory expects (the restricted likelihood removes the degrees-of-freedom loss from estimating the fixed effects). The control (D) agrees with the exact Gaussian case.
- drmTMB REML equals `glmmTMB(REML = TRUE)` to 1e-4 or better, so R's estimator is the joint-Laplace one, as documented.
- `is_converged()` was TRUE in 95% to 100% of REML fits, the same rate as ML in the same cells. I did not inspect the non-converged fits.

What it does not show: that non-Gaussian REML recovers the true SD or is better than ML. Cell B (G = 40) already sits above the truth under both estimators, and cell A overshoots under REML (median 1.156 against 1.0; the medians use 40 data sets, so the Monte Carlo error is large). That matches the `diagnostic_only` class and Julia's own over-correction warning for Poisson. No Julia fit was run.

## 4. Is it one estimator across engines?

Not by default. Inference from the algebra, not measured here. With u integrated by Laplace and a flat prior on beta:

- R (O2): one Laplace over (u, beta). Its restricted term is log det of the full Hessian, which factors as det H_uu times det of the Schur complement S.
- Julia Poisson (Cox-Reid): profile beta out of the Laplace marginal, then subtract half the log determinant of the observed information of that marginal. That information includes the derivative of the half-log-det H_uu term through beta. S does not.

These agree for Gaussian (H_uu does not depend on beta, so both are Patterson-Thompson REML) and differ at third-derivative order for Binomial and Poisson. The two engines would report different variance components on the same data, and neither would be wrong. There is no exact non-Gaussian REML to settle it, because the restricted likelihood has no closed form here. The repo already notes the neighbouring point: drmTMB's AGHQ plus Cox-Reid object "is not asserted to be generic non-Gaussian REML" (`docs/design/224-...md:7-9`). Unverified numerically: I did not code the Cox-Reid objective for Binomial to measure the gap.

## 5. Options weighed

| option | usability (D-139) | statistical meaning | effort |
|---|---|---|---|
| A. Match R's O2 on Binomial | Gives 2 cells and R-script portability for one narrow shape | Valid only if labelled as joint-Laplace; collides with Poisson Cox-Reid under `:REML` | M: add beta to the `:Laplace` route's random block, plus tests, receipts, gap-map rerun |
| B. Keep refusing, clearer message (recommended) | Refusal is already safe; message becomes actionable | No false equivalence; no new unvalidated estimator | S: one string, one test edit |
| C. Document only | Cheapest, but the raw message still misleads | Same as B | XS |

B over C because the present text actively misleads (section 2) and the fix is five lines.

**Proposed message** (keeps "ML-only" for the existing test): "drm (Binomial): `method = :REML` is not available. Binomial is ML-only in DRModels.jl, so omit `method` or pass `method = :ML`. drmTMB fits a binomial REML by a joint-Laplace route that is documented as diagnostic only (docs/design/224), and DRModels.jl does not reproduce it. The Laplace/VA/AGHQ integrator is chosen with `marginal`, not `method`." Apply the same wording to Beta, Gamma and NB2 with the family name substituted.

**If match is wanted later (design for option A).**
1. Compute, on the existing `marginal = :Laplace` binomial route only, a restricted objective: fixed-effect coefficients join the Laplace inner block with a flat prior. Report as `estimation_method = :REML` plus a distinct tag (for example `:REML_joint_laplace`) so Poisson Cox-Reid is never conflated with it.
2. Admit the same shape as R: one ordinary unlabelled `mu` intercept or independent slope, no structured terms, no SD formula.
3. Validate without an exact reference by three anchors: (a) `glmmTMB(REML = TRUE)` and drmTMB `REML = TRUE` on the same data, bar 1e-4 on the SD, as the section 3 table shows is attainable; (b) the Gaussian limit, where the new objective must equal the existing exact Gaussian REML to 1e-8; (c) the direction check from section 3 (REML SD at least ML SD).
4. State in the docs that this is an approximation shared by name with R, not exact REML, and that Poisson `:REML` is a different approximation.

## 6. Unverified

- No Julia fit was run; every statement about Julia is from reading source at `0bfcde2bb`.
- Section 4's claim that joint-Laplace and Cox-Reid differ for Binomial is algebraic, not measured.
- Doc 224's 7.3e-9 and 42% figures are quoted, not reproduced. My own glmmTMB gap is 1e-4 or better, looser because I used default optimizer tolerances.
- The 40-data-set medians are small-sample and say nothing about bias in general.
- drmTMB was read at the committed head of a dirty checkout; the Totoro library is 0.8.1 and may differ from either.
- The effort figures are my estimates.
