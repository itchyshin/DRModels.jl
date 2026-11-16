- **Bare structured intercept-plus-slope terms on Beta and Gaussian REML.**
  `animal(1 + x | g)` and `relmat(1 + x | g)` with no `sd()` formula now fit for
  Beta (ML) and for Gaussian with `method = :REML`, as the two-SD model with
  constant SDs, matching drmTMB 0.8.1.
- **Binomial `animal(1 + x | g)` versus `animal(1 + x | p | g)`.** The
  unlabelled form is now the independent-field model (correlation fixed at 0),
  as in drmTMB 0.8.1; before, it fitted the correlated model, which made its
  log-likelihood differ from R's by 0.9 to 1.4. The labelled form with
  `sd(g, animal, coef(...))` formulas is accepted and fits the correlated model.
