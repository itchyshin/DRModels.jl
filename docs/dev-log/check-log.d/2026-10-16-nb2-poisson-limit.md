# NB2 at the Poisson limit (#1072, #1069, #1039)

- Change: `src/negbinomial.jl` — `_fit_negbin2` and `_fit_negbin2_ranef` now optimise through
  `_nb2_minimize` (AssertionError from the LBFGS/Hager-Zhang line search falls through to
  BackTracking, then Nelder-Mead; a fit on the Poisson plateau, log sigma < -8, is refit once from
  log sigma = -4 and the interior fit is kept only if it has the lower nll) and return through
  `_warn_nb2_poisson_limit` (`src/boundary_diagnostics.jl`), which warns that the fit is at the Poisson
  limit. `converged` is unchanged.
- Tests: `test/test_nb2_poisson_limit.jl` (red on the base commit `727b003`, base + test only).
- Not done: location SEs with phi held at the limit (#1072 request 3) — needs `_vcov_from_hessian`
  (the SE path), left for the #1073 stack. The correlated-slope, zero-inflated and hurdle NB2 fitters
  still call `Optim.optimize` directly.
- Not run locally: Julia is unavailable in this container; verified by CI (see PR body).
