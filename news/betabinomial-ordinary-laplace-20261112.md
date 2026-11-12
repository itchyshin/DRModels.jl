- **`marginal = :Laplace` now works for BetaBinomial with one `(1 | g)`.** It fits the same
  Laplace objective as drmTMB, so the log-likelihood and estimates agree with drmTMB 0.8.1
  (seven simulated data sets: logLik within 1e-4, fixed effects within 1e-3 R SE, log SD within
  5e-3; `test/fixtures/bb_laplace_parity/`, written by `test/parity/gen_bb_laplace_parity.R`).
  The default `marginal = :LA` is unchanged (quadrature) and differs from drmTMB by 0.03 to 0.34
  logLik on the same data. The route covers `sigma ~ 1`, maximum likelihood, no offset; anything
  else is refused by name. `fitted(fit)` on a Laplace fit is still the population value (the
  random-effect modes are not stored yet).
- **Documented:** `CumulativeLogit()` with one `(1 | g)` has no `marginal = :Laplace` option and
  differs slightly from drmTMB (a known engine difference). See the marginal-approximation guide.
