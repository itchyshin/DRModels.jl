- **Binomial quantile residuals no longer throw on two routes.**
  `residuals(fit; type = :quantile)` for a Binomial fit with missing responses
  used to fail with an `InexactError`. It now returns one value per data row, with
  `NaN` at the rows dropped for a missing response, as Poisson and Gamma do.
  The same call on `(1 | g) + (0 + x | g)` used to fail with a `KeyError` because
  that route kept no trial counts. It now stores them, so the call works and
  judges each row against the fixed-effect distribution, as for the other
  crossed and slope shapes.
