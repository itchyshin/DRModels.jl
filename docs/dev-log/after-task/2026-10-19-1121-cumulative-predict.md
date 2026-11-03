# After-task: CumulativeLogit `predict` on new data (#1121)

## 1. Goal

`predict` and `predict_parameters` threw an `ArgumentError` on every
`CumulativeLogit` fit, for any `newdata` including the training data, and with
or without a factor predictor. The message blamed factor levels. The task was
to make prediction return the location linear predictor X·β that drmTMB
returns, without weakening the #1000 column check.

## 2. Implemented

- `src/cumulative_predict.jl` (new, included right after `cumulative.jl`): a
  `_predict_design(::DrmFit{<:CumulativeLogit}, …)` method. It builds the `mu`
  design with the implicit intercept, so factor contrasts are coded exactly as
  at fit time, then drops the `(Intercept)` column, which is the projection
  `src/cumulative.jl` makes after `_design`. The #1000 column check then runs
  on the projected names, with the same message. `predict`,
  `predict_parameters` and their `se = true` variants all reach the design
  through `_predict_design`, so this one method fixes all of them. Every
  ordinal route returns a `DrmFit{CumulativeLogit}`; the fixed and `(1 | g)`
  routes are in the test, and `(0 + x | g)`, `(1 | g) + (1 | h)` and
  `phylo(1 | species)` were checked by hand in this session (predict equals
  X·β̂ exactly on each).
- `test/test_1121_cumulative_predict.jl` with the sweep-oct16 t6 fixtures in
  `test/fixtures/cumulative_predict_1121/` (provenance in its README).
- `news/1121-cumulative-predict.md`; check-log row
  `docs/dev-log/check-log.d/2026-10-19-1121-cumulative-predict.md`.

## 3a. Decisions and rejected alternatives

The fix is a dispatch method in its own file. The smallest edit would have been
a five-line branch inside the generic `_predict_design` in
`src/gaussian_core.jl`; the next most natural home is beside the fitter in
`src/cumulative.jl`. Open PRs that this sweep lane must not overlap change both
files (#1085, #1139 and #1140 change `gaussian_core.jl`; #1080 changes
`cumulative.jl`). Dispatch on `DrmFit{<:CumulativeLogit}` follows the
family-specific projection already in `src/bridge.jl`
(`fit isa DrmFit{<:CumulativeLogit} && param === :mu`). The cost is a
duplicated eight-line #1000 check. Once those PRs land, the method can move
into `cumulative.jl` or be folded into the generic function.

Rebuilding the design from `0 + rhs` was rejected. With no intercept,
StatsModels gives the first factor full dummy coding (`f: a, f: b, f: c`), so
the columns would no longer match the treatment-coded coefficients.

The response scale is unchanged. `_mean_response` is already the identity for
this family, so `type = :response` equals `type = :link`, as in drmTMB (in
`r_cumulative_logit_fixed_pred.csv` the response rows equal the link rows).

## 3b. Evidence

- Red on main (`25964eef1`), Julia 1.12.6: the new file gives 4 passed,
  1 failed, 11 errored; every error is the issue's `ArgumentError`
  ("columns [(Intercept), x, f: b, f: c] but the fitted coefficients are
  [x, f: b, f: c]"), and the failure is the subset-guard message that still
  named `(Intercept)`.
- Green on the branch: 23/23 on Julia 1.12.6 and on Julia 1.10.12.
- drmTMB parity (fixed route, nd_main rows 1:9): max |Δ value| = 2.0e-11,
  max |Δ SE| = 1.0e-8, against drmTMB 0.7.1 re-run locally (R 4.6.0), which
  agrees with the lane's committed `r_cumulative_logit_fixed_pred.csv`.
- The issue's repro script now prints
  `[-0.4209346396520892, 0.0, 0.6314019594781338]` for `predict`,
  `predict(...; se = true)`, `predict_parameters` and its `se = true` variant,
  equal to X·β by hand.

## 4. Verification

Julia 1.12.6, `JULIA_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, Mac Studio.
`Pkg.test()` stops at the first failing file, and it did so here at
`test_bridge_bootstrap_tree.jl` (file 57 of 375). The suite was then run with a
temporary, uncommitted copy of `runtests.jl` that records each file's result
and carries on, in the test environment, under the `Pkg.test()` flags
(`--check-bounds=yes` and the rest), in two shards.

- 372 of 375 files pass, including `test_1121_cumulative_predict.jl`,
  `test_predict.jl`, `test_cumulative.jl`, `test_crossed_cumulative.jl`, the
  `test_cumlogit_*` files, `test_bridge_formula_labels.jl`, `test_aqua.jl`, and
  the JET gate.
- `test_bridge_bootstrap_tree.jl` (Gamma location-scale phylo, 18 pass,
  2 fail at lines 28 and 29) and `test_reml_q4_missing_response.jl` (18 pass,
  1 fail at line 210, `fit_reml.converged`) fail the same way on main
  `25964eef1`, run alone under the same flags.
- `test_crossed_tweedie.jl` did not finish. It ran for 79 min on this branch
  and for 51 min on main, then both runs were stopped. The file fits Tweedie
  crossed random effects and never calls `predict` or `CumulativeLogit`.

No R code changed, so no testthat run or `devtools::document()` applies. This
repository has no `tools/recertify-c17.py` or `tools/capability_ledger.py`.

## 5. Open / follow-ups

- The `predict` docstring says "In-sample, `predict(fit, data) ≈ fitted(fit)`",
  and the `predict_parameters` docstring says the response scale reproduces
  `fit.means[:mu]`. Neither holds for `CumulativeLogit`, where `fitted` is the
  expected category score. The issue asks for a sentence; both docstrings live in
  `src/gaussian_core.jl`, so the sentence is left for after #1085, #1139 and
  #1140 land.
- Draft PR #1100 documents "CumulativeLogit fits are not supported" in the
  `predict` docstring. That line is now wrong and should be dropped there.
- `(1 | g)` route: predictions are X·β̂ as intended, but on this fixture they
  differ from drmTMB's by up to 2.3e-3 because β̂ differs (DRModels' default
  `:LA` integrator is GHQ-32; drmTMB uses the Laplace approximation). The
  difference predates this change and sits in estimation, so the test pins
  X·β̂ on that route and leaves drmTMB's numbers out.
- Crossed `(1 | g) + (1 | h)` ordinal fit (t6 `(1 | g)` fixture plus a
  six-level `h`): every entry of `vcov(fit)` is non-finite, so
  `predict(...; se = true)` returns NaN SEs on that route. The point
  predictions are correct. The non-finite covariance is a fit-level matter
  outside this slice; I did not check whether one of the open J3 SE PRs
  (Refs #1073) covers it.

## 6. Rose audit (claim vs evidence, scope, drift, license)

- Claims: every number above was produced in this session (logs under
  `/tmp/drm1121/`, not committed). No extrapolated numbers.
- Scope: prediction layer only; no fitter, likelihood or estimate changes.
- Drift: new method duplicates the #1000 check text verbatim; noted in 3a.
- License: drmTMB numbers are generated outputs (CSV data and printed
  predictions); no drmTMB source was copied.
