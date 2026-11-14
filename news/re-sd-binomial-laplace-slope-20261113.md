- **`re_sd()` and `vc()` on a Binomial `(1 + x | g)` fit with `marginal = :Laplace`.**
  The slope SD and the covariance were wrong: the fit stores log SDs and an
  atanh correlation, and both functions read them as a log-Cholesky triple.
  They now decode the stored values (on the test data the slope SD was 0.881
  and is now 0.845, the value drmTMB reports). The same fix covers Binomial
  blocks with three or more coefficients, where `re_sd()` now returns one SD
  per coefficient (keys `g_intercept`, `g_x1`, ...). Estimates, standard errors
  and log-likelihoods are unchanged.
