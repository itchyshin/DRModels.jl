- **`fitted(fit)` is now conditional on the bivariate Gaussian route with `(1 | p | g)`.**
  A fit with an ordinary correlated random intercept in `mu1` and `mu2` gives
  `fitted(fit)` as drmTMB 0.8.1 does: the two response means with the group effects
  added. The group effects at the fitted parameters are the exact BLUPs, which the fit
  already kept in `ranef(fit)`. Before, `fitted(fit)` refused on this route (once the
  conditional default was in place) or returned the population values, which differed
  from drmTMB by 0.75 although the log-likelihood agreed to 1e-6. `fitted(fit; re = :zero)`
  still gives the population values. Estimates, log-likelihood and `vcov` are unchanged
  to the last bit. Checked against drmTMB fits (`test/fixtures/fitted_modes_bivariate/`)
  to 1e-6 for the fitted values and the modes, and against modes found by maximising the
  joint density separately in the tests.
  `residuals(fit)` on this route is now `y - fitted(fit)`, conditional on the group effects as
  in drmTMB (it was `y - X b`); `residuals(fit; re = :zero)` gives the old values.
