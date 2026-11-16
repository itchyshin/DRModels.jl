- `fitted(fit)` is now conditional on four more random-effect routes. The
  zero-inflated Poisson, zero-inflated NB2 and zero-truncated NB2 fits with one `(1 | g)`
  on `mu`, and the `CumulativeLogit` fit with `(1 | g) + (0 + x | g)`, keep the
  random-effect modes at the fitted parameters. `fitted(fit)` returns what drmTMB 0.8.1
  returns: `(1 - zi) * mu` (zero-inflated), `mu / (1 - P(0))` (zero-truncated) and the
  expected category score (ordinal), with the modes included in `mu` or in the linear
  predictor. On main these four fits are refused (random effects with these
  families are not yet allowed there). With the random-effect support of #1317
  alone they fit, but `fitted(fit)` is the population value and differs from
  drmTMB by 0.7 to 2.1 although the log-likelihood agrees. With the
  conditional default of `fitted()` also in place, `fitted(fit)` refuses them
  because no modes were kept. Estimates, log-likelihood
  and `vcov` are unchanged to the last bit. `fitted(fit; re = :zero)` still gives the
  population value, and `residuals(fit)` follows `fitted(fit)`. Checked against drmTMB
  fits (`test/fixtures/fitted_modes_zi_ordinal/`) to 1e-6 (2e-5 for the ordinal route),
  and against modes found by maximising the joint density separately in the tests.
