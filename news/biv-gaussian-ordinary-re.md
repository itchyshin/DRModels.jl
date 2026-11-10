- **Bivariate Gaussian with an ordinary correlated random intercept.**
  `bf(mu1 = y1 ~ x1 + (1 | p | g), mu2 = y2 ~ x2 + (1 | p | g), ...)` with
  `Gaussian()` now fits a 2x2 group-level covariance (two SDs and a group
  correlation, reported in the `:phylocov` log-Cholesky block and
  `fit.ranef.Sigma_g`), kept separate from the residual `rho12`. mu1 and mu2
  may have different fixed-effect designs; `sigma1`, `sigma2` and `rho12` may
  have covariates; `method = :REML` is supported; rows with a missing response
  are dropped by default, as in drmTMB. Measured against drmTMB 0.8.1 (ML, REML,
  `rho12 ~ x1`, `sigma1 ~ x1`, missing responses): log-likelihood within 1e-4,
  fixed effects within 1e-3 of the R SE, SEs within 1e-3 relative. Still refused
  with a clear error: random effects on `sigma1`/`sigma2`, one-sided or
  unlabelled `(1 | g)` terms, `(1 + x | p | g)`, partly observed rows
  (`miss_control(response = "include")`), row `weights`, known `V`, and
  Student-t / lognormal bivariate families.
