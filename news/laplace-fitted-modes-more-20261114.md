- **`fitted(fit)` works on two more opt-in Laplace routes (this lifts a refusal).**
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
  The value is the conditional value at the Laplace random-effect modes, not a
  marginal mean. `fitted` and `residuals(fit)` (type `:response`) agree with
  drmTMB 0.8.1 to 1e-6 where both packages use the same Laplace integral
  (`marginal = :Laplace`). On the default (quadrature) `ZeroOneBeta` route they
  differ from drmTMB by up to 6e-4 on the test data, and on the Binomial
  `sd(g) ~ gx` routes by up to 3e-4 on the test data. That gap is the
  difference between quadrature and Laplace. It grows when groups are small:
  with very small or single-member groups the Laplace approximation is
  poorer, so the two integrals disagree more (about 1.3e-3 for the logit link
  and 1.6e-3 for the probit link on one such data set).
  Quantile residuals on these routes are unchanged by this change and still
  differ from drmTMB's: on the `ZeroOneBeta` `(1 | g)` route they judge each
  continuous row against the Beta distribution with the random effect
  integrated out, where drmTMB's are conditional on the modes, and on the
  Binomial `sd(g) ~ gx` routes they are population-level (the modes are not
  used), where drmTMB's are conditional.
