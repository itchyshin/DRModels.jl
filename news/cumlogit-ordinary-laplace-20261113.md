- **`marginal = :Laplace` now works for `CumulativeLogit()` with one `(1 | g)`.** It fits the same
  Laplace approximation as drmTMB, so the log-likelihood, thresholds, fixed effect, log SD and
  standard errors agree with drmTMB 0.8.1 on four simulated data sets (3, 4 and 5 categories; 8 to
  40 groups): log-likelihood within 2e-12, estimates within 1e-9 of a drmTMB standard error,
  category probabilities within 1e-6 (`test/fixtures/cumlogit_laplace_parity/`, written by
  `test/parity/gen_cumlogit_laplace_parity.R`). The default (no `marginal`) is unchanged: it still
  integrates by adaptive quadrature, which sits 0.01 to 0.15 above drmTMB's log-likelihood on the
  same data. Only one ordinary random intercept is accepted; a random slope, a correlated block,
  crossed or structured terms, `sd()` formulas, an offset, a fixed-effects-only model and
  `marginal = :VA` or `:AGHQ` are refused by name. `fitted(fit)` is the expected score at random
  effect zero, as on the default route. `lrtest` between a `:Laplace` and a `:LA` fit is refused;
  bootstrap refits of a `:Laplace` fit use `:Laplace`.
