- **Quantile residuals of a lone mean random slope.** `residuals(fit; type = :quantile)`
  for `y ~ x + (0 + x | g)` read the slope as a random intercept and judged each row
  against the intercept marginal, which is wrong because the slope adds `x^2 sd^2` to the
  variance (a Gaussian example with 40 groups and 800 rows differed by up to 0.84 from the
  correct marginal residual; the SD for `|x| > 1` was 1.20 against 1.01 for the correct
  residual). The fit does not store `x`, so a lone random slope now gets the same
  reference as the other shapes the `residuals` docstring lists (crossed, correlated
  slopes): the random effect is set to 0 and the residuals are over-dispersed under a
  correct model. This is not an accuracy fix: on that example the new residuals are
  further from the correct marginal than the old ones (largest difference 1.44, SD 1.44
  for `|x| > 1`), so do not read them as a marginal check. Random intercepts `(1 | g)`
  are unchanged.
