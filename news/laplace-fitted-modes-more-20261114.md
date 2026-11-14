- **`fitted(fit)` works on two more opt-in Laplace routes.**
  A Binomial fit with a `:probit` or `:cloglog` link and one correlated block
  `(1 + x | g)` made with `marginal = :Laplace`, and a `ZeroOneBeta` fit with one
  random intercept `(1 | g)`, now keep the random-effect modes at the fitted
  parameters. `fitted(fit)` (default `re = :conditional`) and `residuals(fit)`
  return the conditional values, as drmTMB does; before, the call threw because
  no modes were kept. For `ZeroOneBeta` the modes move the logit of the Beta
  component, and the zero and one probabilities stay as fitted. Estimates
  and standard errors are unchanged, and `re = :zero` returns the same
  population values as before. The log-likelihood is unchanged except in the
  last digits (a difference of about 1e-14 on the default `ZeroOneBeta`
  route). The default (adaptive quadrature) `ZeroOneBeta` `(1 | g)` route keeps
  its modes in the same way, so `fitted(fit)` is conditional there too.
  The Binomial random-intercept routes with an SD regression, such as
  `sd(g) ~ gx` with a logit or probit link, share the same fitter. On them
  `fitted(fit)` and `residuals(fit)` now return the conditional values as well,
  where they refused before.
