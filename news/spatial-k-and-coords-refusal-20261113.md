- **`spatial()` given both `K` and `coords` is refused.** A `spatial(1 | g)`
  term (and `sd(g, spatial)`) takes its covariance from `K` or from `coords`.
  When both were passed, the Poisson, NB2 and Beta routes used `K` and dropped
  `coords`, and the bivariate Gaussian route used `coords` and dropped `K`,
  without a word. The call now stops with a message that names both arguments
  and says which single one to give. Fits that give only one of the two are
  unchanged. Gamma already refused this and now shares the same message.
