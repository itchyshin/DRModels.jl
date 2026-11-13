- **Random intercepts with zero inflation, hurdles, zero truncation and a skew-normal
  response now fit, matching drmTMB 0.8.1.** These were refusals ("random effects cannot
  be combined with `zi`/`hu` yet", "supports fixed effects only"). One ordinary `(1 | g)`
  on `mu` is now accepted for `Poisson()` with `zi`, `NegBinomial2()` with `zi` or `hu`,
  `TruncatedNegBinomial2()` (with or without `hu`) and `SkewNormal()`. The fit uses the
  Laplace approximation that drmTMB uses and is tagged `marginal = :Laplace`; it is the
  only integrator on this route, so `marginal = :AGHQ` and `:VA` are refused by name.
  Against drmTMB 0.8.1 on simulated data (`test/fixtures/zi_re_parity/`, written by
  `test/parity/gen_zi_re_parity.R`): logLik within 1e-4, fixed effects within 1e-3 R SE,
  log SD within 5e-3, standard errors within 1e-3 relative.
  Still refused by name: `offset(...)` with these models (not verified against drmTMB),
  random slopes, several random effects, a random effect on `sigma`, `zi`, `hu` or `nu`,
  structured markers, REML, and a hurdle Poisson with a random effect.
  `predict`, `fitted`, `simulate(data = ...)`, `update` and `bootstrap_summary` work on
  these fits. `fitted` is the expected response (zero-inflation and truncation applied);
  `predict` returns the count-component mean, as for the fixed-effects versions.
- **Binomial with three or more crossed random intercepts** `(1 | g) + (1 | h) + (1 | k)`
  fits by the dense Laplace route (drmTMB's objective). Two components keep their own
  sparse route, unchanged. `marginal = :AGHQ` with three components is refused by name.
  These fits are tagged `marginal = :Laplace`; `update(fit, formula; data)` and an explicit
  `marginal = :Laplace` call return the same fit. With exactly two crossed intercepts
  the explicit `:Laplace` keyword is still refused, as before.
- **CumulativeLogit with independent random terms** `(1 | g) + (0 + x | g)` fits by the
  same Laplace route, with the cutpoints as ordinary parameters, so their standard errors
  are reported (the crossed-intercept route reports none). It matches drmTMB 0.8.1 at the
  same bounds. Independent terms with `sd(...)` formulas are refused by name.
- **Zero-inflated, hurdle and truncated NB2 `(1 | g)` fits that stalled now retry from interior
  starts.** The first start is the fixed-effects fit, which can sit on the zero-inflation boundary
  or at the Poisson limit; the fit then stalled up to 10 logLik units below drmTMB (ZINB) or
  returned `NaN`. A fit that did not converge, has no finite logLik or has a zero-mass linear
  predictor of 15 or more is refitted from up to three interior starts, and the refit replaces it
  only if its logLik is higher. A fit that no start rescues keeps `converged = false`. On 120
  simulated sets (30 per family, drmTMB 0.8.1) the fits with no finite logLik went from 10 to 0.
