- **`simulate` and `bootstrap` on a Binomial fit with two crossed random intercepts.**
  `drm(bf(y ~ x + (1 | g) + (1 | g2)), Binomial())` did not keep the number of trials
  on the fit, so `simulate(fit; data)` (and `simulate(fit; re = :zero)`) stopped with
  "does not carry `trials`". The fit now stores the trials (both the 0/1 and the
  `cbind(successes, failures)` form, Laplace and `marginal = :AGHQ`), and the draws match the
  fitted model's own marginal mean and variance. A fit made before this change (which
  stored no trials) still simulates: the trials are rebuilt from the response columns of `data`.
