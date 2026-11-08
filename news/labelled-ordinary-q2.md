- **Ordinary correlated intercept-and-slope with coefficient SD formulas
  (Poisson, NegBinomial2).** `(1 + x | p | g)` (both families) and `(1 + x | g)`
  (Poisson; NB2 already fit it on the two-field route) now fit with
  `sd(g, coef(intercept)) ~ z0` and `sd(g, coef(x)) ~ z1` plus one free
  correlation (`rho = 0.999999 tanh(eta)`, block `:recov`, `"g:atanh_rho"`). It is
  the animal/relmat q2 model with the identity kernel, reusing its Laplace
  helpers. Prediction, the bridge and the marginal simulator accept the form.
  Native drmTMB 0.8.1 refuses the labelled ordinary form and fits only the
  unlabelled `(1 + x | g)` (the same model).
