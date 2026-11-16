- **Non-Gaussian correlated-block shapes that drmTMB 0.8.1 fits now fit.**
  (1) Poisson and NB2 `(1 + x | g)` with an `sd(g, coef(·)) ~ …` formula on only
  one coefficient: the other coefficient keeps a constant SD, as in drmTMB.
  (2) LogNormal `(1 + x | g)` with SD formulas on one or both coefficients (the
  Gaussian correlated-block fit on `log(y)` plus the Jacobian; q = 2 only, as
  drmTMB). (3) Binomial (logit) `(1 + x + z [+ w] | g)` with no SD formula.
  Log-likelihoods agree with drmTMB 0.8.1 to 1e-9 for (1) and (2). For (3) Julia's
  default adaptive Gauss-Hermite log-likelihood is 0.57 (q = 3) above R's Laplace
  value; with `nq = 1` (the Laplace rule) a one-off run gave -253.57472, equal to R's value to 1e-5, but that fit reports `converged = false` and no test calls `nq = 1`, so the claim that the gap is the integrator is not guarded by a test. The q = 4 R fit
  does not converge, so there is no agreement claim there. The post-fit bootstrap
  simulator for the one-target Poisson/NB2 block is not wired.
