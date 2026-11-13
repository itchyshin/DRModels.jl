- **Binomial quantile residuals no longer throw on three routes.**
  `residuals(fit; type = :quantile)` for a Binomial fit with missing responses
  used to fail with an `InexactError`. It now returns one value per data row, with
  `NaN` at the rows dropped for a missing response, as Poisson and Gamma do
  (BetaBinomial too). The same call on `(1 | g) + (0 + x | g)`, `(0 + x | g)`
  alone, or crossed `(1 | g) + (1 | h)` used to fail with a `KeyError` because
  those routes kept no trial counts. They now store them, so the call works and
  judges each row against the fixed-effect distribution, as for the other
  crossed and slope shapes.
