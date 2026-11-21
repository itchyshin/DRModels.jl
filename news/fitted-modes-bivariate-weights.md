- **Weighted fits of the bivariate `(1 | p | g)` route have conditional `fitted()` and `ranef()`.**
  With `weights = w`, a row's weight multiplies its whole log density (as in drmTMB), so the
  group effects are the exact BLUPs under the row precision `w_i * S_i^-1`; a zero weight
  removes that row from its group's mode (a group of zero-weight rows gets a zero effect).
  `fitted(fit)` gives the two response means with these group effects added, as drmTMB does,
  and `fitted(fit; re = :zero)` the population values. `weights` also works together with
  `miss_control(response = "include")`: a row with one response missing keeps its weight on the
  observed cell. Estimates, log-likelihood and `vcov` are unchanged to the last bit. Checked
  against drmTMB 0.8.1 fits on five weighted cells (`test/fixtures/fitted_modes_bivariate_weights/`),
  against modes found by maximising the weighted joint density separately in the tests, and
  against a dense Gaussian log-likelihood written by hand.
