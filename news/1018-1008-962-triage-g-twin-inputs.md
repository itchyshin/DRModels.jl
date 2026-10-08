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
  default to `check_converged = true` (previously `false`). With the default
  `failures = :error`, a replicate that does not converge aborts the bootstrap.
  With `failures = :skip` (and `bootstrap_sigma_a`'s `:warn`), dropped replicates
  stay out of the percentile, `status` is `"bootstrap_incomplete"` rather than
  `"bootstrap"`, and a warning names that status. Retained draws that pile up on
  a random-effect SD or correlation bound set `"bootstrap_at_boundary"` and warn
  as well; a dropped replicate still warns in that case, naming
  `bootstrap_at_boundary`. The q=4 bridge bootstrap uses the same
  `check_converged = true` rule as the univariate bridge. Set
  `check_converged = false` only to reproduce the old intervals.
