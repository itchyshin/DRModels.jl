- **`marginal = :Laplace` now works for `Binomial()` with a `:probit` or `:cloglog` link.** It is accepted
  for one `(1 | g)` and for one correlated block `(1 + x | g)` on the mean (including `cbind(s, fl)`
  responses), and fits the same Laplace approximation as drmTMB. On four simulated cells (probit and
  cloglog, each with `(1 | g)` and with `(1 + x1 | g)`) the log-likelihood agrees with drmTMB 0.8.1 to
  1e-8 or better, and the fixed effects and log SDs to 1e-8
  (`test/fixtures/laplace_more_shapes_parity/`, written by
  `test/parity/gen_laplace_more_shapes_parity.R`). The correlation of the probit `(1 + x1 | g)` cell is on
  its boundary in both engines and is not compared. The default (no `marginal`) is unchanged: it still
  integrates by adaptive quadrature, which sits 0.002 to 0.03 above drmTMB's log-likelihood on these
  data. Before, every non-logit link was refused under `marginal = :Laplace`. A random slope alone,
  offsets, `sd()` formulas, `marginal = :VA` and REML are still refused. `fitted(fit)` is the
  population-level value, as on the default route. `lrtest` between a `:Laplace` and a `:LA` fit is
  refused; bootstrap refits of a `:Laplace` fit use `:Laplace`.
