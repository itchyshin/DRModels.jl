- **`marginal = :Laplace` now works with a `sigma` formula for Gamma, Beta, NegBinomial2 and ZeroOneBeta.**
  One ordinary `(1 | g)` on the mean with fixed-effect covariates on `sigma` (for example `sigma ~ x1`) is
  fitted by the same Laplace approximation as drmTMB. On one simulated data set per family the
  log-likelihood agrees with drmTMB 0.8.1 to 1e-4 or better, and the fixed effects and log SD to 5e-3 or
  better (`test/fixtures/laplace_sigma_formula_parity/`, written by
  `test/parity/gen_laplace_sigma_formula_parity.R`). `ZeroOneBeta()` takes the `marginal` keyword for the first
  time: `:LA` (the default, unchanged) or `:Laplace` for one `(1 | g)`; other values are refused by name.
  The default (no `marginal`) is unchanged and sits 0.01 to 0.06 above drmTMB's log-likelihood on these data.
  A random effect on `sigma`, offsets together with a `sigma` formula, random slopes and `sd()` formulas stay
  refused. `lrtest` between a `:Laplace` and a `:LA` fit is refused; bootstrap refits of a `:Laplace` fit
  use `:Laplace`.
