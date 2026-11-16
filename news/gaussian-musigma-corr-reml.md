- **REML for the Gaussian labelled pair `(1 | p | g)` in `mu` and `sigma`.**
  `drm(bf(y ~ x + (1 | p | g), sigma ~ 1 + (1 | p | g)), Gaussian(); method = :REML)`
  now fits (before, it was refused although drmTMB 0.8.1 fits it). It maximises
  drmTMB's objective, one Laplace approximation over the group effects and both
  the `mu` and `sigma` coefficients, so it is an approximate restricted
  likelihood. On three datasets it matches drmTMB's REML log-likelihood within
  2e-9 and coefficients within 2e-5 R standard errors. The ML route and its
  defaults are unchanged.
