- **Quantile residuals of a lone mean random slope.** `residuals(fit; type = :quantile)`
  for `y ~ x + (0 + x | g)` read the slope as a random intercept and judged each row
  against the intercept marginal, which is wrong because the slope adds `x^2 sd^2` to the
  variance (a Gaussian example differed by up to 1.5 from the correct marginal residual).
  The fit does not store `x`, so a lone random slope now gets the same reference as the
  other shapes the `residuals` docstring lists (crossed, correlated slopes): the random
  effect is set to 0 and the residuals are over-dispersed under a correct model. Random
  intercepts `(1 | g)` are unchanged.
