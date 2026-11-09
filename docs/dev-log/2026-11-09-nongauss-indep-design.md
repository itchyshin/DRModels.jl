# Independent intercept-plus-slope terms on the non-Gaussian mean routes, design and pre-registration, 2026-11-09

Reader: whoever owns the R-Julia parity lane. Purpose: say what is being added, what "agrees with R" means, and what is out of scope, before any reference number is read.

## Gap

The wave-10 gap map (`gap-map-wave10-20261109`, tip `0bfcde2bb`) lists about 40 cells where drmTMB 0.8.1 fits `y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) [+ (0 + x2 | g) [+ (0 + x3 | g)]]` and DRModels.jl refuses it with "supports multiple random effects only as crossed/nested intercepts". Gaussian, Gamma and Student already take this shape.

## Change (phase 1, no SD formulas)

1. `src/indep_slopes_laplace.jl`: the dense TMB-convention Laplace fitter `_fit_indep_re_laplace` (already used by Gamma and Student) gets per-family observation terms (value, first and second derivative in the linear predictor, expected second derivative) for Poisson, NB2, Binomial (logit) and Beta. It also works with no dispersion block (Poisson, Binomial).
2. `src/poisson.jl`, `negbinomial.jl`, `binomial.jl`, `beta.jl`: the "multiple random effects" branch calls it when the terms are all `(1 | g)` or `(0 + x | g)` and at least one is a slope. Correlated `(1 + x | g)` beside another term stays refused. `marginal = :AGHQ` and `:VA` stay refused. REML stays refused (non-Gaussian REML is ML-only here).
3. `src/lognormal.jl`: `log(y)` is Gaussian, so independent terms delegate to the exact Gaussian multi-component marginal (the route crossed intercepts already use), with the Jacobian shift.
4. Tweedie: decided after the other families (needs a profile or joint search over the power `p`); see "Per-cell verdicts" in the PR.

## Pre-registered agreement rule

Reference: drmTMB 0.8.1, `engine = "tmb"`, Totoro library `~/drm_campaign_20261108/Rlib`, the gap-map data (`data_<family>_ordinary.csv`, 400 rows, 39 groups), `y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) ...` with `sigma ~ 1` where the family has one. Same data and formula in Julia.

A cell **agrees** when all hold at the optimum:

- `|logLik_Julia - logLik_R| <= 1e-4` (the gap-map threshold);
- every fixed-effect coefficient (mean and, where present, log-sigma) differs by at most `1e-3` (absolute, log or logit scale as the models define them);
- every random-effect log SD differs by at most `5e-3`.

The coefficient and SD limits are looser than the logLik limit on purpose: at a flat optimum a logLik gap of 1e-4 can come with a parameter gap of a few 1e-3. They are not to be widened after the results are read. A cell that misses one of them is investigated under rule (c) below, not accepted.

Per-cell verdicts after the run:

- (a) fits and agrees;
- (b) R itself does not converge (R `is_converged() = FALSE` or non-positive-definite Hessian): record, leave refused unless Julia agrees with R's own number;
- (c) Julia stops at a worse optimum than R: add restarts only if a restart demonstrably reaches R's logLik;
- (d) too large: record the effort.

## Scope guard

- Gaussian SD gaps, Gaussian REML and non-converged triage belong to other lanes and are not touched.
- No tolerance in an existing test is changed. Tests whose refusal wording changes are updated to the new wording.
- Julia-side changes are in a draft PR only. The parity verdicts in drmTMB flip later in a separate reviewed PR.

## Phase 2 (SD formulas on independent terms)

Decided after phase 1 and recorded in the PR: the same agreement rule, with the references taken from the gap-map R runs (`sd(g, coef = ...) ~ ...`).

## Result (written after the run; the rule above was not changed)

Reference runs on Totoro: 136 cells (112 `indep`/`slope`, 24 crossed intercepts `(1 | g) + (1 | g2)` as a control for the shared SD hooks). drmTMB 0.8.1 fits 96 and errors on 40 (beta 10, tweedie 10, lognormal 8, student 8, gamma 4: every `sd()`-on-independent-terms cell of those families except the single-slope ones). Julia fits all 96 reference cells. The committed tool `tools/nongauss_indep_compare.jl --all` tabulates them; the subset kept as a test is `test/test_nongauss_indep_routes.jl`.

- **(a) fits and agrees: 75 of 96.** All `sd-none` cells (poisson, nb2, binomial q2 to q4, beta, lognormal, gamma, student, tweedie q2 to q4), all crossed-intercept controls, and every SD-formula cell on independent terms of Poisson (q2 to q4, `one`/`z`/`first`/`last`), NegBinomial2 (q2 to q4), Gamma (q2) and Binomial (q2 `last`, q3 `first`). The worst gaps over the 74 cells other than Binomial q4 are logLik 7.4e-9, fixed effects 1.5e-6 and log SD 3.2e-7, against limits of 1e-4, 1e-3 and 5e-3. Binomial q4 `sd-none` agrees on logLik (1.5e-9) and fixed effects (5.5e-11) and on the three non-zero SDs, but its fourth SD is at zero (log SD -17.0 in Julia, -10.4 in R; drmTMB flags the fit itself), so only its SD criterion is waived; the test says so explicitly.
- **(b) drmTMB errors or is itself not clean: 40 errored cells, plus 5 flagged.** The 40 stay as they are: Beta and Tweedie independent terms with `sd()` formulas stay refused (as in R); Gamma q3/q4, Student and LogNormal independent terms with `sd()` formulas fit in Julia, with no R number to compare. The 5 flagged cells (`is_converged() = FALSE`) are Binomial `sd-first` q2 and q4 and `sd-z` q2, q3, q4 (the last three also belong to (e)).
- **(c) Julia at a worse optimum than R: Binomial `sd-first` q2 and q4 (0.44 and 0.45 logLik below R's).** R's value comes from an intercept SD going to zero on every group (`sd` coefficients -54 and +57, `is_converged() = FALSE`). From four alternative starts and BFGS/L-BFGS the Julia fit stays on the ridge at log SD about (-14.3, +15.7), because beyond about 20 the Laplace Hessian loses positive definiteness and the objective returns the failure value. The restart pattern of `_q2_boundary_restarts` does not reach R's value, so it was not added. Julia reports `converged = false` for both cells, as R does.
- **(d) too large: none.** Slowest cells: Tweedie q2 at 94 s (series density in dual numbers), Tweedie q3 and q4 at about 57 s; the old Binomial complete-SD route needs 62 to 566 s.
- **(e) 19 cells where the existing single-term and complete-SD routes disagree with R's Laplace by 0.03 to 0.99 logLik, route unchanged.** These are `slope` cells with `sd-one`/`sd-z` on Poisson, NegBinomial2, Binomial, Beta, Gamma and Student, the three Tweedie slope cells (`sd-none`, `sd-one`, `sd-z`; the Tweedie slope route is GHQ and unchanged), and Binomial q2 `sd-one`, q2 to q4 `sd-z`. They run the older per-group quadrature routes (`test_iid_sd_quadrature_accuracy.jl` checks them against an exact integral), whose logLik differs from the Laplace value drmTMB reports (higher in the Beta, Binomial q2 `sd-one` and q2 `sd-z` cases I checked by hand; the sign was not tabulated for the rest); the same models with no formula (`sd-none`, my route) agree with R to 1e-10. Nothing in this PR touches those routes. `binomial|indep|q2|sd-one` is the same model as `sd-none` and R gives the same number for both (-255.7796), while the existing route returns -255.4850; sending it through `_drm_indep_sd_laplace` reproduces R exactly. Whether those routes should switch to the Laplace objective for drmTMB parity is a decision for the owner of the SD-regression routes.

## Shapes outside the reference

`(1 | g) + (0 + x | h)` (a slope on a different grouping factor than the intercept) now also reaches the same fitter on every family that takes this route. It uses the same per-component code as the same-factor case but has no drmTMB reference in this PR.

## Gap-map rerun (all 3008 cells, Julia only)

Harness `docs/dev-log/gap-map-wave10-20261109/` (branch `claude/gap-map-wave10-20261109`): `driver.py julia 8 300 cells_run.tsv` on Julia 1.10.12, Totoro, this branch; R side = the stored wave-8 drmTMB 0.8.1 results (not rerun); `classify.py`, then `compare10.py` with the wave-10 map (tip `0bfcde2bb`) as base. Tables in `docs/dev-log/2026-11-09-nongauss-indep-gap-map-rerun/`; in them `w8_*` columns are the base (wave-10 tip) and `w10_*` columns are this branch.

| class | base | this branch |
|---|---|---|
| both-fit-agree | 260 | 309 |
| both-fit-disagree | 54 | 54 |
| R-only | 107 | 57 |
| Julia-only | 43 | 59 |
| both-refuse | 2538 | 2522 |
| nonconverged | 5 | 6 |
| crash (either side) | 0 | 0 |

- **49 flips to agreement:** 48 R-only to both-fit-agree and 1 nonconverged to both-fit-agree (`gamma|indep|q2|sd-none`). By family: poisson 14 (11 indep, 2 crossed with `sd()`, 1 slope), nb2 10, binomial 6, lognormal 5, beta 4, tweedie 3, gamma 4 (incl. the nonconverged one), student 3 (2 crossed with `sd()`, 1 slope). No R-only cell of kind `indep`, `slope` or `crossed` (ML) is left.
- **16 both-refuse to Julia-only** (gamma q3/q4, lognormal and student independent terms with `sd()`; drmTMB errors on all of them).
- **2 R-only to nonconverged:** Binomial `sd-first` q2 and q4 (verdict (c) above). Julia now returns an honest `converged = false` fit where it refused before.
- **No regression:** `got_worse.tsv` is empty and no previously fitting cell moved by more than 1e-6 in logLik. The 54 both-fit-disagree cells are unchanged (the older quadrature routes, verdict (e)).
- A first full-suite run on Julia 1.10.12 found one regression of my own, fixed before this rerun and before the numbers above: `_indep_sd_shape` called `_split_ranef`, which throws its own message for `animal`/`relmat`/`spatial` slope forms, so the NB2 structured SD tests (9 files) errored instead of reaching their route; a missing grouping column surfaced as an `ErrorException` instead of `ArgumentError`.
