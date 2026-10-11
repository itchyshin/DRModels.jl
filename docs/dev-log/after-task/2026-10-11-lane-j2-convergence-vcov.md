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

- **Coordinator follow-ups.** #935 is the #1047 defect (fixed there). #961
  (vcov half) and #968 share `_vcov_from_hessian_checked`: a non-finite Hessian
  on finite data gives a NaN vcov and `converged = false` (Student fixed / RI /
  correlated-RE fitters, Gaussian fixed). #952: a fit ending on the
  failed-objective sentinel warns at fit time. #978: the sparse phylo-mean
  route adds the β–variance cross block (`G Vvv Gᵀ`), so ML mean SEs come
  from the joint observed information.
- **Test correction.** `test_bridge.jl` asserted a finite lower `resd` profile
  bound that was the #1023 sentinel artefact; it now accepts a genuine
  crossing above the domain edge or an open (-Inf, "did not cross") arm.

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
- Parked from the follow-ups: #936 (same recalibration as #1049), #937
  (Student nu multi-start), the line-search half of #961 and the tree-scale
  root cause of #952.
- #1049, #1050 and #983 describe code that exists only on draft PR #1044
  (`_unitfree_converged`); nothing to change on main until it merges.

## 4. Verification

Julia could not be installed in the authoring container, so nothing was run
locally. PR CI run 38112128451 on head c699354: all 8 test shards (Julia 1.10
and 1, 4 shards each) and `ci-ok` green. Round 1 (run 38100089811) failed on
the bridge sentinel assertion and on a wrong #956 repro; both corrected.

## 5. Rose

Claims are limited to what the regression file asserts. No drmTMB source was
vendored. No bridge payload shape changed; the value of `diagnostics.converged`
now matches the top-level bridge `converged` (is_converged) for degenerate fits.
