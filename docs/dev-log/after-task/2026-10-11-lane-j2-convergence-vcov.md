# After-task: lane J2 — convergence, optimiser and vcov

## 1. Goal

Work the lane-J2 issues from the 2026-10 review campaign: places where a
convergence flag, a covariance or a confidence-interval endpoint reports success
that the code did not establish. Shannon, with Noether (engine, convergence
flags) and Rose (claim versus evidence). No subagents.

## 2. Implemented

- **#1024** The five sparse-Laplace inner mode searches (`sparse_laplace_glmm.jl`)
  return `ok = false` when the iteration cap is exhausted without meeting `tol`;
  callers already take the 1e18 sentinel path on `!ok`. The coordinate-spatial
  Poisson mode (`poisson.jl`) does the same, with a 1e-8 relative rounding-floor
  allowance because its `tol = 1e-12` can sit below the floor of an
  ill-conditioned `Cinv`.
- **#1047** `em_fit` returns `converged` (its relative log-likelihood stop fired);
  the `algorithm = :em` adapter uses it instead of `iterations < 500`.
- **#964** The temporal two-start comparisons use `_better_restart` (objective
  re-evaluated at each minimizer).
- **#956** The remaining raw-inverse Wald sites (bivariate REML, Gaussian REML,
  two-structured Gaussian, slope-axis RE, cross-family rho CI) go through
  `_vcov_from_hessian` / `_vcov_from_fd_hessian`. The spatial-Poisson and
  phylo-interaction sites named in the issue were already guarded on main.
- **#968** `_fit_fixed_gaussian` reports a non-finite Hessian on finite data
  (constant response) as a NaN vcov plus `converged = false` and a warning,
  instead of the LAPACK `ArgumentError`. `_vcov_from_hessian` itself still throws
  on a NaN Hessian (owner decision 14; NaN-predictor fits rely on it).
- **#949** `show`, `check_drm().converged` / `ok` and `bridge_diagnostics().converged`
  use `is_converged(fit)`.
- **#1003** Failed variance/temporal boundary diagnostics warn with the error
  (`_boundary_diagnostic_failed`) instead of returning `nothing` silently, at
  fit time and inside `check_drm`.
- **#1005** `_fit_locscale` records the optimiser used, logs why a fallback ran,
  and certifies a Nelder-Mead result with an exact-gradient check.
- **#1007** `heritability` / `icc` results carry `status` and warn on a
  non-converged or MAP fit, a non-finite delta SE, or a profile edge.
- **#1023** The location-only `resd` profile path treats a sentinel/non-finite
  objective or an unconverged nuisance solve as a failed arm, reports a profile
  still below the threshold at the `|log sd| < 50` domain bound as unbounded,
  and computes `failed` / `*_endpoint_failed` / `*_nuisance_*` instead of
  hard-coding them.

## 3a. Decisions and rejected alternatives

- The `heritability` CI stays clamped to `[0, 1]` when the SE is not finite
  (existing tests assert the clamp); `status = :vcov_unavailable` says so.
  Returning NaN bounds is a contract change left to Shinichi.
- `check_drm().ok` is still unscored against the gradient when no gradient
  exists (`grad_source = :none/:unavailable`); that is a documented contract.
- Parked for decision: #969 (pseudo-inverse sends a flat coordinate to ~0
  variance; changing it touches ~95 call sites), #946 (q4/q2/coevolution
  `f_reltol` stops on the verified engine), #965 (VA inner stationarity
  detector needs a place on the fit).
- #1049, #1050 and #983 describe code that exists only on draft PR #1044
  (`_unitfree_converged`); nothing to change on main until it merges.

## 4. Verification

Julia could not be installed in the authoring container, so nothing was run
locally. The PR's GitHub Actions CI is the test runner; see the PR for the run id.

## 5. Rose

Claims are limited to what the regression file asserts. No drmTMB source was
vendored. No bridge payload shape changed; the value of `diagnostics.converged`
now matches the top-level bridge `converged` (is_converged) for degenerate fits.
