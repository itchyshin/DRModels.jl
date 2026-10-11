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

- Added after the first push (#927, #953, #945), all of which move fits or
  messages that parity receipts may pin:
  - #927 `src/gaussian_ranef.jl`: the 2-D mode finder of the Gaussian mean +
    sigma random-intercept route (`_fit_musigma_ranef_gaussian`) now accepts a
    step only if it lowers the Laplace objective h(b, δ) (Armijo), with Newton on
    a positive-definite-shifted Hessian; a non-PD Hessian returns a finite
    barrier. Fits that already converged to a minimum are unchanged to
    optimiser tolerance; cells that used to hit a saddle or a DomainError now fit.
  - #953 `src/cumulative.jl`: the crossed ordinal route starts its cutpoint
    search from the best of the empirical start and each single-random-intercept
    fit's cutpoints, and restarts Nelder–Mead until no improvement. logLik can
    only move up for the same data (the `ordinal::wine` cell: -81.7577 →
    drmTMB's -81.5654).
  - #945 `src/sparse_laplace_glmm.jl`: 12 internal "currently supports a
    constant sigma formula" errors (crossed + relmat/animal/spatial NB2, Gamma,
    Beta Laplace/AGHQ) are user-facing; the six crossed fitters now also refuse
    a non-constant single column (`sigma ~ 0 + x`).

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

- #1000 residual gap: on the routes with no fit-time schema (bivariate and
  non-Gaussian families) the guard compares coefficient NAMES. An unseen level
  that sorts before every fitted non-reference level, while the fitted reference
  level is absent from `newdata`, yields the same names and is not detected.
  Closing that needs the training levels stored on every route (decision 3 in
  the PR).
- #770 clash: the NB2 `(1|g)` route now soft-clamps η_σ (identity in ±20,
  smooth beyond); #770's hard-clamp reported sigma should be reconciled when both
  merge (reported scale = `exp(softclamp(η_σ))` there).

## 4. Verification

Julia cannot be installed in the authoring container; the suite was run only in
the PR's GitHub Actions CI. See the PR for the run id.
