- **Confidence level and bootstrap replicate count are checked (#1018).**
  `confint`, `coeftable`, `profile_result`, `profile_curve`, `bootstrap_ci` /
  `bootstrap_summary` / `bootstrap_result`, `bootstrap_sigma_a`, `profile_sigma_a`,
  `bias_correct`, and `heritability` / `icc` / `repeatability` reject a `level`
  outside `(0, 1)` or a non-finite `level`, and bootstrap rejects `B < 1` or a
  non-integer `B`, with an error that names the argument and the value. `level = -0.5`
  no longer returns an inverted interval, and `level = 95` no longer surfaces as a
  bare `DomainError` from `quantile`. The cross-family `fit_mixed_family` route
  uses the same `level` check; its `B = 0` still means "do not bootstrap".

- **`update` repeats the estimator stored on the fit (#1008).** A REML fit stays
  REML, a non-default `marginal` stays that integrator, and a penalized MAP fit
  keeps its `penalty`. Keywords still override those stored options. An unnamed
  extra positional argument is now an error. Options that were never stored on
  `DrmFit` (`K`, `A`, `tree`, `coords`, `algorithm`, `g_tol`, `profile_ci`,
  `phylo_coupled`, `sparse`, `impute`, `missing`) are not recovered; pass them
  again. Bootstrap refits of a MAP seed forward the same stored penalty.

- **Behaviour change: parametric bootstrap now drops unconverged replicates by
  default (#962).** `bootstrap_ci`, `bootstrap_summary`, and `bootstrap_result`
  default to `check_converged = true` (previously `false`). A refit that does
  not converge is skipped: it stays out of the percentile, `status` is
  `"bootstrap_incomplete"` rather than `"bootstrap"`, and a warning names that
  status. The default `failures = :error` aborts only when a refit throws.
  Retained draws that pile up on a random-effect SD or correlation bound set
  `"bootstrap_at_boundary"` and warn as well; a dropped replicate still warns
  in that case, naming `bootstrap_at_boundary`. Gaussian location-scale,
  random-intercept, and location-scale-scale fits judge `converged` by the
  Newton step in standard-error units (`max |H⁻¹g| / SE ≤ 1e-3`) together with
  `Optim.converged` and a stop short of the iteration limit. The absolute
  `g_converged` check at `1e-8` is not part of that flag, so a change of
  predictor units does not turn a stationary fit into a failed one. The q=4
  bridge bootstrap uses the same
  `check_converged = true` rule as the univariate bridge. Set
  `check_converged = false` only to reproduce the old intervals.

- **`update` and the likelihood-ratio tests refuse silent mismatches (#1008, #1002).**
  `update(...; method = :ML)` on a penalized MAP fit drops the stored penalty
  unless `penalty` is passed again. Passing `family` to `update` is an error
  that names the argument. `lrtest` and `lrt_boundary` error when the two fits
  have different `nobs`, different marginal approximations, or a penalized MAP
  estimate, so a missing-response or imputed refit is not compared with the
  seed as if they were the same sample.
