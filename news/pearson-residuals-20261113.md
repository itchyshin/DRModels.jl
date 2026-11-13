- **`residuals(fit; type = :pearson)` is new.**
  It returns `(y - E[Y]) / sqrt(Var[Y])`, with the mean and variance of the fitted distribution of each observation.
  It is implemented for fixed-effect Gaussian, Poisson, NegBinomial2, Gamma and Beta fits, and matches drmTMB 0.8.1 `residuals(fit, type = "pearson")` to 1e-6 on one cell per family.
  Any other family or route throws an `ArgumentError` that names it: random effects, structured or temporal terms, zero inflation, hurdles, Student, LogNormal, Binomial, ordinal and bivariate fits.
  For some of these drmTMB defines the Pearson residual differently, for example on the log scale for LogNormal, so no number is returned until the definition is checked.
  `type = :response` and `type = :quantile` are unchanged.
