# mf_profile_far_point

Committed data for `test/test_mixed_family_profile_far_point.jl` (n = 150,
columns `y1`, `y2`, `x`). Generated in R (so it does not depend on the Julia
version's RNG) with the same recipes and seeds as drmTMB's
`tests/testthat/test-xfam-bridge.R` cross-family live round-trips:

- `gaussian_poisson.csv`: `set.seed(20260610)`; `x`, `u` = `rnorm(150)`;
  `y1 = 0.5 + 0.8 x + 0.7 u + rnorm(150, sd = 0.5)`; `y2 = rpois(150, exp(0.4 + 0.3 x + 0.4 u))`.
- `gamma_poisson.csv`: same seed and `x`, `u`; `y1 = rgamma(150, shape = 2, scale = exp(0.3 + 0.2 x + 0.4 u) / 2)`; `y2` as above.
- `nb2_gaussian.csv`: `set.seed(20260611)`; `y1 = rnbinom(150, size = 3, mu = exp(0.4 + 0.3 x + 0.4 u))`;
  `y2 = 0.5 + 0.8 x + 0.7 u + rnorm(150, sd = 0.5)`.
