# After-task: FX2 simulate newdata + LogNormal marginal= (sweep-oct16 t6)

## 1. Goal

Close the two API gaps on lane t6 (pins drmTMB `ca3ddefa5`, DRModels.jl
`f71b292`): Julia `simulate` has no `newdata` (twin of drmTMB #1536), and
`drm(..., LogNormal())` has no `marginal=` (Poisson/Gamma/NB2/Binomial/Beta/Student
do). Shannon, with Noether (engine) and Hopper (twin surface). No subagents.

## 2. Implemented

- `src/gaussian_core.jl` `simulate`: optional `newdata` rebuilds a shallow
  `DrmFit` from `predict_parameters` so every existing `_simulate_once` family
  branch draws one row per newdata row. Training-row call unchanged. Temporal
  fits refuse `newdata`. Auxiliary slots (`:trials`, `:beta_mu`, `:ordinal_*`)
  are recycled when constant or taken from newdata.
- `src/lognormal.jl` `drm`: `marginal::Symbol = :LA`. `:Laplace` on ordinary
  `(1 | g)` tags the existing AGHQ-K=1 fitter (already exact for identity-link
  log y). Other symbols, and `:Laplace` on FE / slope / crossed / structured,
  throw `ArgumentError`.
- Tests: `test/test_fx2_simulate_newdata.jl`, `test/test_fx2_lognormal_marginal.jl`.

## 3a. Decisions and Rejected Alternatives

- Did **not** add LogNormal to `_drm_ordinary_laplace` (Poisson/Binomial/NB2/Gamma/Beta
  TMB-convention spine). The LogNormal (1|g) path is already exact; a second
  integrator would be a methods choice.
- `newdata` is accepted and used (the issue's first expected), not a named unused-arg
  error. drmTMB #1536 still ignores `newdata`; this is the Julia twin doing the
  useful half.

## 4. Verification

See the PR body for the red-proof transcript, the green re-run, and the full
suite result. C17 not recertified (no drmTMB `R/` / `src/` change).

GitHub issue write returned 403; this PR Refs itchyshin/drmTMB#1536 and
#942 / #1071. Never Closes/Fixes.
