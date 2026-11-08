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

## Results (read after the thresholds above were committed)

Files: `test/fixtures/tweedie_structured_sd/agree_r.tsv`, `agree_julia.tsv`; summary by `agreement_compare.py`.
R drmTMB 0.8.1 (Totoro library, not verified which commit), Julia at this branch, same CSV files. 10 fits (5 datasets x
sd-none, sd-z), 55 coefficients. R took 0.7 to 1.5 s per fit; Julia 5 to 17 s per fit (Mac, one thread, includes first-call compilation for the first fit).

| Quantity | Threshold | Result, all 10 fits | Result without `phylo_1`/sd-z |
| --- | --- | --- | --- |
| median `d` | <= 0.01 | 1.6e-08 pass | 1.4e-08 pass |
| 99th percentile `d` | <= 0.05 | 0.75 **fail** | 5.6e-07 pass |
| max `d` | <= 0.08 | 0.78 **fail** | 5.6e-07 pass |
| median `d / SE_R` | <= 0.05 | 2.0e-07 pass | 9.2e-08 pass |
| max `d / SE_R` | <= 0.25 | 4.30 **fail** | 1.4e-06 pass |
| non-finite Julia SE | 0 | 0 pass | 0 pass |
| non-finite R SE | 0 | 5 **fail** (see below) | 5 **fail** |
| boundary flags | 0 | 0 pass | 0 pass |
| Julia converged, R `convergence == 0`, R `pdHess` | all | all true pass | all true pass |

Per fit, max `d`: nine fits between 1.6e-08 and 5.6e-07; `phylo_1`/sd-z 0.78.

**The one disagreement (`phylo_1`, sd-z), and what it is.** R reports convergence ("Newton polish; max |gradient| = 4e-9") at a
different point from Julia: mu intercept -0.52 vs 0.22, SD intercept -0.29 vs -0.88, SD slope on z -0.13 vs 0.65. Evaluating each
engine's own objective at both points (a within-engine check, not a likelihood comparison across engines): Julia's objective is
285.686 at the R point and 282.978 at the Julia point; R's objective gives the same two numbers, 285.686 and 282.978. So the two
engines evaluate the same objective at the same parameters, and R stopped at a stationary point that is 2.7 nats worse than the one
Julia found. This is a local optimum in R's fit of this dataset, not a Julia error and not a model difference. It does fail the
pre-registered agreement threshold and is reported as a failure; I did not drop it. I did not search for other starts in R, so I
cannot say whether R's optimiser would find the better point from another start.

**R SE count.** The 5 non-finite R SEs are the single log-SD coefficient of each sd-none fit. I took R's standard errors from
`vcov(fit)` aligned to `fit$opt$par`, and for the bare `phylo()`/`relmat()` model R's covariance matrix gives NA for that coefficient. I did not
establish whether this is R's reporting or my extraction, so the "R SE non-finite" count is a failure of the protocol as written,
not a Julia finding. Julia's SE for the same coefficient is finite in all 5 fits.

## What changed in the code

- `src/tweedie.jl`: a bare `phylo(1 | g)` or `relmat(1 | g)` mean intercept becomes the SD route with `sd(g, …) ~ 1`
  (same device as the Gamma/count q2 default). `sd(g, phylogenetic) ~ …` is a new SD target on the same Laplace fitter the relmat route uses,
  with the tip-correlation precision from `_structured_q2_kernel` (ultrametric check, every tip observed).
- The inner latent-mode solve now uses the exact structured gradient and Hessian (`Qsd * b` plus per-observation scalar eta
  derivatives) in a damped Newton, instead of ForwardDiff over the whole G-vector. Same estimates (relmat `sd ~ 1`, 40 groups: max difference
  3e-9 across all 7 coefficients), 84 s to 21 s including compilation. This is what lets 40-group fits finish in 5 to 20 s; the gap map's
  `relmat sd-z` timeout was at 300 s.
- Refusal wording: the old "structured covariance inputs ... no random-effect SD target" and "no structured route for tweedie() (matches drmTMB)"
  messages were wrong after this change and are replaced; `sigma`/`nu` refusals now name the structured kind.

## Still refused, and why

- Structured slopes, q >= 2, labelled blocks: R drmTMB 0.8.1 refuses these for Tweedie too (gap map rows 4 to 10).
- `animal` and `spatial` Tweedie: R refuses all 50 cells of each (gap map `_KW_R_GRID`).
- `sigma`/`nu` predictors with a structured intercept: not in the R grid; left refused, untested against R.
- Structured intercept combined with an ordinary random effect: not in the R grid.
- Cross-engine log-likelihood or AIC: never compared.
