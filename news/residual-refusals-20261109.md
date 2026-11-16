- **Bare structured terms without an `sd()` formula.** Binomial `animal(1 | g)`,
  `relmat(1 | g)` and `spatial(1 | g)`, Beta `spatial(1 | g)` and
  `spatial(1 + x | g)`, and Beta `phylo(1 + x | g)` now fit as the model with
  constant SDs, the same fit as the explicit `sd(g, level) ~ 1` form and the
  one drmTMB 0.8.1 fits. The labelled NegBinomial2 `phylo(1 + x | p | g)` with no
  SD formula fits as well. drmTMB flags its Binomial spatial and Beta spatial
  q = 2 fits as not converged on the gap-map data; the log-likelihoods agree.
