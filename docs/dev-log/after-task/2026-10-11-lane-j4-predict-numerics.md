# After-task: lane J4 — predict, residuals and density numerics

## 1. Goal

Work the lane-J4 code-review issues of the 2026-10-11 campaign: #1028, #998,
#982, #1039, #1015, #1006, #1000, #1016, #999, #995, #996, #981, #959.

## 2. Implemented

- `src/tweedie.jl`: the Dunn–Smyth series window starts at the Stirling peak,
  is polished to the integer maximum, and grows until terms fall 40 nats below
  it; NaN (never a finite truncated value) past 1e6 terms per side (#1028).
- `src/negbinomial.jl`: plain, ZINB and `(1|g)` / `(1+x|g)` NB2 objectives use
  `_nb2_logpmf` (no quantised p = r/(r+μ)); the `(1|g)` routes use the existing
  `_softclamp` instead of a hard clamp on η_σ; `_nb2_loggammadiff`'s middle branch
  uses `loggamma(k) - logbeta(r, k)` (#998, #982, #1039).
- `src/gaussian_core.jl`: `_design` names a missing column, a `missing` value or
  an unseen level before `modelcols` and never stores a newdata schema
  (`store_schema = false`); `predict` / `predict_parameters` apply a fitted
  Poisson `offset()` and refuse a newdata design whose columns differ from the
  fitted coefficient names (#1015, #1006, #1000).
- `src/summary.jl`: `coeftable` docstring now states the working-scale null of
  every block (#1016).
- `src/variational.jl`: NB2/Gamma/Beta VA kernels require a constant sigma
  column; stored scale is the clamped value the objective scored, with a warning
  when the clamp is active (#999, #959 diagnostic).
- `src/quantile_residuals.jl`: NB2 conditional distribution switches to its
  Poisson limit once p would quantise; a structured mean random intercept warns
  that the σ_b-marginal PIT may be mis-calibrated (#995, #996, #981).
- `src/sparse_laplace_glmm.jl`: non-Gaussian phylo routes give the tree-height
  warning (#996).
- Tests: `test/test_lane_j4_predict_numerics.jl` (auto-discovered).

## 3a. Decisions and Rejected Alternatives

- #995/#996: a correct structured PIT needs either conditional residuals or the
  per-unit variance σ_b²·C_ii (tree/K not stored on the fit). Left as an owner
  decision; a warning replaces the silent mis-calibration.
- #1000: a schema cache on `BivariateDrmFormula` was not added: the cache is
  keyed on the formula object, so refitting one formula to data with another
  level set would reuse the first schema. The coefficient-name guard removes the
  silent case for every family instead.
- #959: sparse-Laplace and Tweedie clamps left in place (owner decision on
  widening to drmTMB's ±12 soft band vs a `check_drm` flag).

## 4. Verification

Julia cannot be installed in the authoring container; the suite was run only in
the PR's GitHub Actions CI. See the PR for the run id.
