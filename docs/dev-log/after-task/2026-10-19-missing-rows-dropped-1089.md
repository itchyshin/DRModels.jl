# After-task: `drm` drops rows with a missing predictor or response (#1089)

Shannon speaking, with the Noether (engine entry) and Pat (warning and error text)
perspectives. No subagents ran.

## 1. Goal

A `missing` predictor made `drm` throw `KeyError: key missing not found` from the
StatsModels schema, and a missing response on the Gaussian `(1 | g)` route was refused
with a route-level `ArgumentError`. drmTMB drops those rows (R's `na.omit`). Owner
decision for this lane: drop rows with a missing predictor or response, with a warning
that gives the count, and do not stop with a named error.

## 2. Implemented

New helpers in `src/missing_data.jl`:

- `_drop_missing_predictor_rows(f, data)` drops the rows with a `missing` value in a
  predictor column (fixed-effect terms of every parameter, plus random-slope
  covariates; grouping and structured-marker columns are excluded). It warns with the
  count and the column names and returns the subset table. Kept `Union{Missing,T}`
  vectors are narrowed to `T`, so a numeric predictor is not dummy-coded; `Vector{Any}`
  columns (R factors from the bridge) keep their type. `mi(x)` formulas are untouched.
- `_drop_missing_response_rows(f, data)` does the same for the response.
- `_check_drop_keeps_levels` refuses a drop that would change the first-appearance
  level sequence of a `relmat`/`animal`/`spatial` grouping column, because those
  routes match their matrix to the levels by position. `phylo` matches by tip label.
- `_drop_scope_weight_rows!` subsets prior `weights` (#931, from #1061) with the rows.

Call sites: the shared non-Gaussian `_fit_observed_response_rows` wrapper (every
non-Gaussian univariate family and the ordinary-Laplace route), the Gaussian fitter,
SkewNormal, and the four bivariate entries (Gaussian, LogNormal, Student, mixed-family
pair). On the Gaussian path, missing responses are dropped before the design is built
when the model has a `(… | g)` random effect or `meta_V` and no structured, temporal or
`sd()` term (the routes that used to refuse). The other Gaussian missing-response
routes are unchanged. The `bootstrap_ci` length error now names `drm_listwise`.

## 3a. Decisions and rejected alternatives

- Drop, not refuse (owner decision). Earlier draft #1132 named the error instead and
  was closed unmerged.
- Rows are dropped from the data table before any design is built, so group indexes
  and the training schema come from the kept rows. Dropping inside the route after the
  design (the #482 trap) was rejected.
- `NaN`/`Inf` predictors are not dropped: they are a separate input error (#1113,
  draft #1045 rejects them). Missing grouping labels are not dropped either (#1112,
  draft #1137 refuses them). Under the same owner decision, #1112 would be a drop;
  that is for the owner to settle between the two PRs.
- Missing responses on `relmat`/`animal`/`spatial` and the phylo dense fallback stay
  refused. Those refusals are pinned by existing tests, and the positional matching
  makes a silent misalignment possible.
- `bootstrap_ci` on the raw data still fails its length check (the seed fit has fewer
  rows than `data`). Making the bootstrap redo the drop is follow-up work. Only the
  message changed.

## 3b. Evidence

- New `test/test_1089_missing_rows_dropped.jl`, 51 assertions: the issue's Gaussian
  `(1 | g)` call, Gaussian fixed effects, prior weights (Gaussian, Poisson), Poisson and
  Gamma `(1 | g)`, bivariate Gaussian, Student and LogNormal, the `(Gaussian, Poisson)`
  pair, SkewNormal, `meta_V`, the `relmat` level-order guard, the bootstrap message
  and the all-missing refusal. Each fit equals the fit on `drm_listwise` data exactly
  (`theta ==`). On `main` 25964eef1 the same file gives 2 pass, 3 fail, 45 errors (the issue's
  `KeyError` and the route-level refusal); on this branch 51 of 51 pass.
- `test/test_lss_missing_response.jl`: the two assertions that pinned the old refusal
  on `(1 | g)` and `meta_V` now check the new contract (count in the warning, `nobs`,
  `theta` equal to the hand-dropped fit). The `relmat` and phylo-hetero refusals are
  unchanged.
- drmTMB parity on the 27 raw datasets of the t5 sweep (drmTMB repo,
  `sweep-oct16/t5`, commit ba1460ad; Gaussian, Poisson and Gamma `(1 | g)` with 0, 5
  and 20% missing in `y` and `x`; Poisson and Gamma with `marginal = :Laplace`): 27 of
  27 fits returned, `nobs` equal to native drmTMB in 27 of 27, max |ΔlogLik| 9.7e-11,
  max |Δβ_μ| 6.5e-10. The summary bar for this check was 7 of 27 on `main`.
- Full suite, every `test_*.jl` file in its own process (`--check-bounds=yes`, one
  BLAS and one Julia thread), Julia 1.12.6 on Totoro, run on this branch (b4ec3cb8b)
  and on `main` 25964eef1 side by side. Per-file outcomes are identical except the new
  file. Assertions over the files both runs finished: `main` 13,032 pass, 1 fail; branch
  13,085 pass, 1 fail (+53 = the new file's 51, plus 2 from the rewritten LSS block).
  `test_crossed_tweedie.jl` hit the 60-minute per-file cap on both trees (exit 124),
  as in draft #1137's runs; CI's sharded run is the authority for it.

## 4. Pre-existing failures, not caused by this change

- `test_reml_q4_missing_response.jl`: `fit_reml.converged` fails at line 210 on `main`
  and on this branch alike (18 pass, 1 fail on both). Also reported in draft #1137.
- `test_crossed_tweedie.jl` is the slow file: see the evidence above.

## 5. Open / follow-ups

- `bootstrap_ci(fit; data)` with the raw data after a drop (workaround:
  `drm_listwise(formula, data)` as `data`).
- #1111: a `Union{Missing,Float64}` predictor with no missing value is still
  dummy-coded when nothing is dropped.
- #1112 (missing grouping label) and #1113 (`NaN` predictor) are separate, with their
  own drafts.
- Missing responses on the structured Gaussian routes stay refused.

## 6. Rose audit

Not done. Left for review. License boundary intact: no drmTMB source was read or
vendored; parity uses drmTMB's committed output CSV only.
