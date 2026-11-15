- **Random effects on `sigma` beyond a Gaussian single intercept.** Eight shapes now fit and match drmTMB 0.8.1
  (log-likelihood within 1e-4): `sigma ~ x1 + (1 | g)` for LogNormal, Gamma, NegBinomial2 and BetaBinomial (fixed-effect
  mean); Gaussian `sigma ~ x1 + (0 + x1 | g)`, `sigma ~ x1 + (1 + x1 | g)` (one correlation parameter),
  `sigma ~ x1 + (1 | g) + (1 | h)`, and a mean `(1 | g)` with a sigma `(1 | h)` on a different group. All use the
  Laplace approximation with constant SDs (no `sd()` formula). `simulate`, `bootstrap_result`, `update`, `predict`,
  `re_sd` and `ranef` work on them. Other combinations (a mean random effect with a sigma one for the count and
  positive families, sigma slopes outside Gaussian, offsets) still refuse by name.
- **Fix: quantile residuals and `repeatability` on a sigma random slope.** A Gaussian `sigma ~ x + (0 + x | g)` fit stores
  its SD as `g_x_logsigma`, which was not read as a scale-axis effect. `residuals(fit; type = :quantile)` then
  integrated out a random intercept on the mean that the model does not have, and `repeatability` and the variance
  boundary diagnostic counted the sigma slope as a mean variance component. All three now treat it as a scale-axis effect;
  quantile residuals on these fits use the stored population mean and sigma.
