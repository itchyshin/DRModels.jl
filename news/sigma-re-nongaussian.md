- **Random effects on `sigma` beyond a Gaussian single intercept.** Eight shapes now fit and match drmTMB 0.8.1
  (log-likelihood within 1e-4): `sigma ~ x1 + (1 | g)` for LogNormal, Gamma, NegBinomial2 and BetaBinomial (fixed-effect
  mean); Gaussian `sigma ~ x1 + (0 + x1 | g)`, `sigma ~ x1 + (1 + x1 | g)` (one correlation parameter),
  `sigma ~ x1 + (1 | g) + (1 | h)`, and a mean `(1 | g)` with a sigma `(1 | h)` on a different group. All use the
  Laplace approximation with constant SDs (no `sd()` formula). `simulate`, `bootstrap_result`, `update`, `predict`,
  `re_sd` and `ranef` work on them. Other combinations (a mean random effect with a sigma one for the count and
  positive families, sigma slopes outside Gaussian, offsets) still refuse by name.
