- **Gaussian `sd(g, sigma) ~ …` with a sigma random effect (dense Laplace).** A
  mean random intercept `(1 | id)` with `sd(id) ~ w` can now sit beside a sigma
  random intercept `(1 | site)` or slope `(0 + z | site)` with
  `sd(site, sigma[, coef(z)]) ~ v`, on different (even crossed) groups, and a
  sigma-only `sd(g, sigma) ~ …` fits. `sigma ~ 1 + animal(1 | id)` /
  `relmat(1 | id)` also fits with `sd(id, animal|relmat, sigma) ~ z`. Both use
  the TMB-convention Laplace marginal (twins of drmTMB's
  `test-gaussian-mu-sigma-joint-sd.R` and `test-gaussian-sigma-structured-sd.R`),
  checked against an independent dense Laplace at the fitted parameters
  (|logLik| difference below 1e-4). Maximum likelihood only, at most 600 latent
  coordinates; phylo and spatial sigma effects are refused. Design and risks:
  `DESIGN-joint-sigma.md`.
