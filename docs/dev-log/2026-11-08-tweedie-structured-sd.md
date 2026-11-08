# Tweedie structured (phylo / relmat) mean intercept: comparability with R and agreement thresholds

Date: 2026-11-08. Branch `claude/tweedie-structured-sd-20261108` (from `claude/wave6-rtfix-20261108`).

## The gap

The refusal-gap map (`docs/dev-log/parity-crosswalk-evidence-20261107/fuzz-gap-map` in the drmTMB
evidence worktree) lists these Tweedie cells where R drmTMB 0.8.1 fits and DRModels.jl refused:

| Cell | R model | Julia before |
| --- | --- | --- |
| `tweedie|phylo|q1|sd-none` | `phylo(1 | g, tree = tree)` | refused |
| `tweedie|phylo|q1|sd-one` | plus `sd(g, level = "phylogenetic") ~ 1` | refused |
| `tweedie|phylo|q1|sd-z` | plus `sd(g, level = "phylogenetic") ~ z` | refused |
| `tweedie|relmat|q1|sd-none` | `relmat(1 | g, K = K)` | refused |
| `tweedie|relmat|q1|sd-z` | `sd(g, level = "relmat") ~ z` | fit existed, timed out (300 s) |

R refuses every structured Tweedie slope or q >= 2 cell, so those stay refused here too.

## Can R's Laplace and Julia's integration be compared honestly?

For these routes, yes at the optimum, because the approximation is the same one.

- R (TMB): the structured intercept is a G-dimensional latent vector with precision `Q = C^-1 / sd^2`;
  the marginal likelihood is the Laplace approximation (random effects `b` integrated out).
- Julia: the existing related-matrix Tweedie route (`_fit_tweedie_relmat_sd_laplace`) is also a Laplace
  approximation over the same G-dimensional vector. The adaptive Gauss-Hermite rule is used only for the
  ordinary one-dimension-per-group routes. This task adds no quadrature.
- Both engines evaluate the compound Poisson-Gamma density with the Dunn-Smyth series, so the observation
  density is the same up to series truncation.

So both engines maximise the same Laplace objective, and differences at the optimum come from optimiser
tolerance and series truncation. This is different from the ordinary-intercept Tweedie route, where Julia's
adaptive Gauss-Hermite and R's Laplace give estimates that differ by about 0.03 to 0.08 and the maximised
log-likelihoods are not comparable.

Rules used here regardless:

- Compare **estimates at each engine's own optimum**, on the link scale (`mu` coefficients, `log sigma`,
  `nu` on the logit-(1,2) scale, SD coefficients on the log scale).
- **Never compare log-likelihood or AIC across engines.** The constants dropped by each engine's Laplace
  code differ and the maps found 56 both-fit cells whose log-likelihoods disagree.
- The kernel scale is checked, not assumed: phylo uses the tip correlation `ape::vcv(tree, corr = TRUE)`
  (as the Julia q2 phylo route does, checked there against R); relmat uses `K` raw.

## Agreement protocol (thresholds fixed before any R-vs-Julia result was read)

Five simulated datasets (committed in `test/fixtures/tweedie_structured_sd/`, 40 groups x 8 rows), fitted by
R drmTMB 0.8.1 (`engine = "tmb"`, Totoro library) and by Julia on the same CSV. Models: phylo sd-none,
phylo sd-z, relmat sd-none, relmat sd-z (the datasets differ in seed, kernel, and true parameters).

Per fitted coefficient (matched by name and position), with `d = |theta_julia - theta_R|`:

| Quantity | Threshold |
| --- | --- |
| median `d` over all coefficients and datasets | <= 0.01 |
| 99th percentile of `d` | <= 0.05 |
| max `d` | <= 0.08 |
| median `d / SE_R` | <= 0.05 |
| max `d / SE_R` | <= 0.25 |
| non-finite Julia SE count | 0 |
| non-finite R SE count | 0 |
| boundary flags (|log sigma| >= 15, |nu logit| >= 12, any SD coefficient with |value| >= 6, either engine) | 0 |
| Julia `converged` and R `convergence == 0` and `pdHess` | all true |

A failed threshold is reported as a failure, not loosened. A dataset where either engine does not converge
is reported and counted, not dropped silently.
