# #995: Gaussian `phylo(1 | g)` quantile residuals, DRModels.jl vs drmTMB

Date: 2026-10-16 · lane: d3 (PR #1078, Refs #995) · harness:
`tools/parity_phylo_quantile_residuals.R` (+ its Julia half
`tools/parity_phylo_quantile_residuals.jl`) →
`docs/dev-log/evidence/parity-phylo-quantile-residuals.tsv`

## Builds compared

| | DRModels.jl | drmTMB |
|---|---|---|
| Commit | `dbe2d45224d8b162d8b3afef6c42e76aa074f925` (PR #1078 head; `src/` clean) | `ca3ddefa5ea11bbfd3b765615dfbb89305d2485a` (origin/main, installed from a scratch clone, not vendored) |
| Version | 0.7.2 + PR #1078 | 0.7.1, code hash `7c5b5e9f40f7faafd82b89b980ea0c9e171ac3ba08f547ae5b0835d009cec977` |
| Runtime | Julia 1.12.7 (conda-forge) | R 4.5.3, TMB 1.9.23 (conda-forge) |
| Call | `drm(bf(@formula(y ~ x + phylo(1 \| species)), @formula(sigma ~ 1)), Gaussian(); data, tree)` | `drmTMB(bf(y ~ x + phylo(1 \| species, tree = tree), sigma ~ 1), family = gaussian(), data = dat)` |
| Residual | `residuals(fit; type = :quantile)` | `residuals(fit, type = "quantile")` |

Both fits are ML (each engine's default).

## Result

| Case | n / tips | σ̂_phylo / σ̂ | max \|q_julia − q_drmTMB\| | engine gap, drmTMB's convention | engine gap, DRModels.jl's convention | logLik (Julia / drmTMB) | converged |
|---|---|---|---|---|---|---|---|
| `fixture_h1` (test/parity gaussian-phylo-mean data, height-1 tree) | 72 / 18 | 2.97 | **1.457** | 3.4e-09 | 2.6e-09 | −32.73529897 / −32.73529897 | Julia `true`; drmTMB code 0, `is_converged` TRUE, pdHess TRUE |
| `fixture_hraw` (same data, raw tree, h = 2.474) | 72 / 18 | 2.97 | **1.457** | 6.1e-12 | 7.6e-11 | −32.73529897 / −32.73529897 | same |
| `strong_signal` (seed 995, rcoal(120), 3 rows/tip) | 360 / 120 | 6.11 | **3.929** | 9.6e-10 | 1.3e-09 | −249.88068809 / −249.88068815 | same |

- "Engine gap, drmTMB's convention" is max |(y − fitted_julia)/σ̂_julia − q_drmTMB|. "Engine gap, DRModels.jl's convention" is max |q_julia − (y − Xβ̂_drmTMB)/√(σ̂_drmTMB² + σ̂_phylo,drmTMB²)|, with C_ii = 1 on drmTMB's normalised tree.
- Coefficients agree to ≤ 1.0e-10 and σ̂ to ≤ 2.0e-10. σ̂_phylo agrees to ≤ 3.0e-09 on drmTMB's scale (`re_sd(fit; scale = :drmtmb, tree)`). The fixture logLik matches the committed drmTMB 0.7.0 value (−32.735298972267131).
- drmTMB's quantile residual equals its own `(y − fitted(fit)) / sigma(fit)` to 1.1e-15 in every case.

## Reading

**The two engines differ by design, not by estimation.** Under either convention the engines agree to about 1e-9, roughly the optimiser tolerance. The headline differences (1.46 and 3.93) are the convention:

- **DRModels.jl (PR #1078):** the population-level PIT, (y − Xβ̂) / √(σ̂_e² + σ̂_b² C_ii).
- **drmTMB 0.7.1:** the conditional residual, (y − Xβ̂ − û_tip) / σ̂. `predict()` adds the phylo modes when `newdata` is NULL (`R/methods.R:2995-3001`, read in the scratch clone), and `fitted_distribution()` builds the CDF from that prediction.

**drmTMB's documentation disagrees with its code.** `R/adequacy.R:8-14`, `R/methods.R:3816-3831` and `NEWS.md:2664-2669` describe these residuals as "at the fixed-effect, population-level parameter estimates … conditional on the fixed-effect prediction, not marginal". That wording is accurate only when `newdata` is supplied. On the fitted rows the code conditions on the modes.

**This is queued as a drmTMB-side owner decision.** Should drmTMB's quantile residual for structured-mean fits be the population-level PIT (what DRModels.jl now returns) or the conditional one (what its code does)? Whichever it is, should its docs say so? Nothing in drmTMB was edited. This is not drmTMB#1513, which an earlier PR body cited: #1513 is a draft about `mi()` quadrature.

**An interpretive caveat for either choice.** Population-level residuals of related tips are correlated under the model. On `fixture_h1` the mean between-row correlation is 0.54, so a calibrated PIT has an expected sample SD of 0.68 (0.80 on `strong_signal`), not 1. The observed SDs are 0.43 and 0.87. Conditional residuals are shrunk instead (SDs 0.93 and 0.89 here). Neither should be read against N(0, 1) row by row without that adjustment.

## A finding on the way: the quadrature was not exact

PR #1078's head `8563231` computed the same population-level PIT from a 32-node Gauss–Hermite mixture. Against the closed form now used, that mixture is off by 2.5e-4 on the fixture and by **0.035** on `strong_signal` (column `max_abs_q_gh32_vs_closed`). The quadrature error grows with σ̂_b √C_ii / σ̂_e:

| ratio | 1 | 2 | 2.5 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|
| max residual error, \|q\| ≤ 4 | 2e-11 | 1.3e-6 | 5.4e-5 | 5.6e-4 | 6.2e-3 | 0.020 | 0.12 | 0.21 |

Commit `dbe2d45` evaluates the Gaussian marginal in closed form on this route. The same 32-node mixture is still used for an ordinary Gaussian `(1 | g)` mean intercept, so that route carries the same error when σ̂_b / σ̂_e is large. That is out of scope here and is listed as a follow-up in the after-task report.

## Reproduce

```
DRMTMB_GIT_SHA=<drmTMB commit> Rscript tools/parity_phylo_quantile_residuals.R
```

Run it from the DRModels.jl root, with drmTMB installed and `julia` on PATH. The run is deterministic: a second run reproduced every number above.
