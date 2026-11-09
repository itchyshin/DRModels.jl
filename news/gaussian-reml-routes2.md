- **Gaussian `method = :REML` on more routes.** REML now fits a correlated block of
  3 or 4 coefficients `(1 + x1 + x2 [+ x3] | g)`, a correlated block of 2 to 4
  coefficients beside crossed intercepts `(1 + x1 | g) + (1 | g2)` (labelled or
  not), `relmat` / `animal` / `spatial(1 | g)` with its own `sd(g, <level>) ~ 1` or
  `~ z`, and the bare unlabelled `relmat` / `animal` / `spatial(1 + x | g)`. The
  ordinary, crossed and `sd(g, <level>)` shapes agree with drmTMB 0.8.1
  `REML = TRUE` on the datasets where R converged (REML logLik to 1e-4, mean
  coefficients within 0.02 R standard errors, SDs and correlations within 1e-2);
  on ordinary q = 4 that is two datasets of three, because neither fit converges
  on the third. The bare `relmat` / `animal` / `spatial(1 + x | g)` cells have no
  R reference of their own: they are completed to the explicit `sd() ~ 1` form,
  and the tests only check that the two forms give the same fit. The labelled
  forms were compared with R on one shape (ordinary q = 3). The evidence is in
  `docs/dev-log/2026-11-09-gaussian-reml-routes2.md`. These shapes keep a constant
  `sigma` and no `sd()` formula on the correlated blocks (drmTMB refuses REML there
  too). The bare `spatial(1 | g)` stays ML only: it estimates the exponential range
  where drmTMB fixes it, so the maximum likelihoods differ by design; the refusal now
  says so and points to `sd(g, spatial) ~ 1`, which supports REML.
