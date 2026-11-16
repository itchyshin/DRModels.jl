- **`fitted(fit)` works on an opt-in Laplace fit of one ordinary `(1 | g)`.**
  A fit made with `marginal = :Laplace` on a Poisson, NegBinomial2, Binomial,
  Gamma or Beta mean with a single random intercept now keeps the random-effect
  modes at the fitted parameters, so `fitted(fit)` (default `re = :conditional`)
  and `residuals(fit)` return the conditional values, as on the default route
  and as drmTMB does. Before, the default call threw because no modes were
  kept. The log-likelihood, estimates and standard errors are unchanged
  (the modes are stored after the fit, not used in it), and `re = :zero`
  returns the same population values as before. `ranef`, `predict` and
  `simulate` give the same results and errors as before: they do not read the
  stored modes. A Poisson `(1 | g)` fit with prior weights still refuses
  `marginal = :Laplace`.
