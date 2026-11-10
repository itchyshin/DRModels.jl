- **Independent-term Laplace fits follow a ridge in the group SD coefficients.** The Laplace
  objective for independent `(1 | g) + (0 + x | g)` terms (Gamma, Student, Beta, NB2, Poisson,
  Binomial with `sd()` on some terms) returned its failure value when any group log SD left
  `[-30, 30]`. On a monotone ridge, for example `sd(g, coef(intercept)) ~ z` on Binomial data,
  drmTMB follows the coefficients to about -54 and 57 (group log SD near -111), and Julia stopped
  at the wall with `converged = false` and a log-likelihood 0.44 below drmTMB's. The bound is now
  300, so Julia reaches drmTMB's optimum (log-likelihood within 1e-8, converged). Fits that never
  touched the old bound are unchanged.
