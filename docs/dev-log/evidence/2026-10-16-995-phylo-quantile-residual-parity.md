# #995: Gaussian `phylo(1 | g)` quantile residuals, DRModels.jl vs drmTMB

Date: 2026-10-16 · lane: d3 (PR #1078, Refs #995) · harness:
`tools/parity_phylo_quantile_residuals.R` (+ its Julia half
`tools/parity_phylo_quantile_residuals.jl`) →
`docs/dev-log/evidence/parity-phylo-quantile-residuals.tsv`

Terminology. "Marginal" below means the distribution of y with the phylo effect integrated out: N(x'β̂, σ̂² + σ̂_b² C_ii). PR #1078's NEWS calls this "population-level". drmTMB uses "population-level" for something else: its fixed-effect-only `newdata` prediction, with σ̂ alone. That is why this note does not use the word across engines.

## Builds compared

| | DRModels.jl | drmTMB |
|---|---|---|
| Commit | `9a86d58cde3e11fce2f29de01f859f89370acbfb` (PR #1078; `src/` clean). The residual code is as in `dbe2d45`; `9a86d58` only changes `ranef()`'s return type. | `ca3ddefa5ea11bbfd3b765615dfbb89305d2485a` (origin/main), installed from a scratch clone, not vendored |
| Version | 0.7.2 + PR #1078 | 0.7.1, code hash `7c5b5e9f40f7faafd82b89b980ea0c9e171ac3ba08f547ae5b0835d009cec977` |
| Runtime | Julia 1.12.7 (conda-forge) | R 4.5.3, TMB 1.9.23 (conda-forge) |
| Call | `drm(bf(@formula(y ~ x + phylo(1 \| species)), @formula(sigma ~ 1)), Gaussian(); data, tree)` | `drmTMB(bf(y ~ x + phylo(1 \| species, tree = tree), sigma ~ 1), family = gaussian(), data = dat)` |
| Residual | `residuals(fit; type = :quantile)` | `residuals(fit, type = "quantile")` |

Both fits are ML (each engine's default).

## Result

| Case | n / tips | σ̂_phylo / σ̂ | max \|q_julia − q_drmTMB\| | engine gap, drmTMB's convention | engine gap, DRModels.jl's convention | logLik (Julia / drmTMB) | converged |
|---|---|---|---|---|---|---|---|
| `fixture_h1`: test/parity gaussian-phylo-mean data, height-1 tree | 72 / 18 | 2.97 | **1.457** | 3.4e-09 | 2.6e-09 | −32.73529897 / −32.73529897 | Julia `true`; drmTMB code 0, `is_converged` TRUE, pdHess TRUE |
| `fixture_hraw`: same data, raw tree (h = 2.474) | 72 / 18 | 2.97 | **1.457** | 6.1e-12 | 7.6e-11 | −32.73529897 / −32.73529897 | same |
| `strong_signal`: seed 995, rcoal(120), 3 rows/tip | 360 / 120 | 6.11 | **3.929** | 9.6e-10 | 1.3e-09 | −249.88068809 / −249.88068815 | same |

- **Engine gap, drmTMB's convention:** max \|(y − fitted_julia)/σ̂_julia − q_drmTMB\|.
- **Engine gap, DRModels.jl's convention:** max \|q_julia − (y − Xβ̂_drmTMB)/√(σ̂_drmTMB² + σ̂_phylo,drmTMB²)\|. C_ii = 1 on drmTMB's height-normalised tree.
- **Estimates:**
  - coefficients agree to ≤ 1.0e-10 (9.2e-11);
  - σ̂ to ≤ 2.1e-10;
  - σ̂_phylo to ≤ 3.1e-09, on drmTMB's scale via `re_sd(fit; scale = :drmtmb, tree)`.
- **Fixture logLik** matches the committed drmTMB 0.7.0 value, −32.735298972267131.
- **drmTMB's residual is conditional.** It equals its own `(y − fitted(fit)) / sigma(fit)` to 1.1e-15 in every case.

## Reading

**The two engines differ by design, not by estimation.** Under either convention they agree to about 1e-9, roughly the optimiser tolerance. The headline differences (1.46 and 3.93) come from the definition of the residual:

| | residual | DRModels.jl vs it (max \|Δ\|) |
|---|---|---|
| DRModels.jl (PR #1078) | marginal PIT, (y − Xβ̂) / √(σ̂_e² + σ̂_b² C_ii) | — |
| drmTMB 0.7.1, **its code** | conditional, (y − Xβ̂ − û_tip) / σ̂ | 1.457 / 1.457 / 3.929 |
| drmTMB 0.7.1, **its residual docs** | fixed-effect only, (y − Xβ̂) / σ̂ ("not marginal") | 1.814 / 1.814 / 7.521 (29 of 360 rows ±Inf on `strong_signal`) |

drmTMB's code takes the conditional route. `predict()` adds the phylo modes when `newdata` is NULL (`R/methods.R:2995-3001`, read in the scratch clone). Its `?predict` page documents this: "When `newdata = NULL`, predictions are for the fitted rows and include currently implemented conditional random-effect contributions for `mu`, including … phylogenetic `mu`" (`R/methods.R:2858-2862`).

**drmTMB's residual documentation describes the third row instead.** Each source, quoted verbatim:
- `R/methods.R:3816-3818` (`?residuals`): "`qnorm(F(y; theta_hat))` at the fitted, fixed-effect distributional parameters".
- `R/methods.R:3827-3829`: "for random-effect or structured fits, these residuals are conditional on the fixed-effect prediction, not marginal".
- `R/adequacy.R:8-12`, a code comment, not user-facing: "evaluates F at predict_parameters()'s FIXED-EFFECT, population-level parameter estimates. For a fit with random effects or other structured components this makes the residual conditional on the fixed-effect prediction, not marginal on the full model".
- `NEWS.md:2664-2666`: "for random-effect or structured fits, quantile residuals are conditional on the fixed-effect prediction, not marginal".

The harness computes that documented residual with drmTMB's own `qnorm(fitted_distribution(fit, newdata = dat)$p(y))`. It equals (y − Xβ̂)/σ̂ to ≤ 4e-15 on the fixture. On `strong_signal`, `pnorm` saturates and 29 rows come back ±Inf. The finite tail rows also lose up to 0.078 in the `qnorm(pnorm(·))` round trip.

**Queued as a drmTMB-side owner decision; nothing in drmTMB was edited.** For structured-mean fits, which residual should `residuals(type = "quantile")` return:
1. conditional (what the code does);
2. fixed-effect only (what the residual docs say);
3. marginal (what DRModels.jl now returns)?

This is tracked as twin-track issue #1101, for mirroring on drmTMB's tracker. Whichever it is, the docs and the code should then agree. Only option 3 gives parity with DRModels.jl: "make the code match the docs" (option 2) would still differ by up to 1.8 on the fixture. This is not drmTMB#1513, which an earlier PR body cited; #1513 is a draft about `mi()` quadrature.

**What this comparison does not test.** In every case the tree is ultrametric and every tip is observed, so C_ii is the same for every row. drmTMB's `phylo()` aborts on non-ultrametric trees and normalises tip variance to 1, so the cross-engine check cannot reach the per-tip σ_b²·C_ii part of the change (#996). `fixture_hraw` does check a constant C_ii = h ≠ 1. The tip-varying case is covered only by DRModels.jl's own tests: `test/test_995_tip_variance_linear.jl` (caterpillar tree, C_ii from 0.15 to 7.05) and `test/test_995_phylo_pit_centre.jl`.

**An interpretive caveat for any choice.** Marginal residuals of related tips are correlated under the model, so even a calibrated PIT should not be read against N(0, 1) row by row.
- **Plug-in value.** On `fixture_h1` the fitted model's mean between-row correlation is 0.54, which gives √E[s²] = 0.68 (0.80 on `strong_signal`). This treats the fitted V as the truth and ignores the estimation of β and the variances.
- **Parametric bootstrap.** 200 seeded refits at the fitted parameters, simulating y from N(Xβ̂, V):

  | case | sd(q) 5–95% band | median | observed sd(q) | percentile of observed |
  |---|---|---|---|---|
  | `fixture_h1` | 0.47–1.02 | 0.69 | 0.43 | 2.5th (low, but within the bootstrap's support) |
  | `strong_signal` | 0.45–1.23 | 0.68 | 0.87 | 77th |

- **Conditional residuals are shrunk instead**: their SDs are 0.93 and 0.89 here.

## A finding on the way: the quadrature was not exact

PR #1078's head `8563231` computed the same marginal PIT from a 32-node Gauss–Hermite mixture. Against the closed form now used, that mixture is off by 2.5e-4 on the fixture and by **0.035** on `strong_signal` (column `max_abs_q_gh32_vs_closed`). Its error grows with σ̂_b √C_ii / σ̂_e:

| ratio | 1 | 2 | 2.5 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|
| max residual error, \|q\| ≤ 4 | 2e-11 | 1.3e-6 | 5.4e-5 | 5.6e-4 | 6.2e-3 | 0.020 | 0.12 | 0.21 |

Commit `dbe2d45` evaluates the Gaussian marginal in closed form on this route. The ordinary Gaussian `(1 | g)` mean intercept still uses the same 32-node mixture, so it carries the same error when σ̂_b / σ̂_e is large. That is out of scope here and is tracked in #1103.

## Reproduce

```
DRMTMB_GIT_SHA=<drmTMB commit> Rscript tools/parity_phylo_quantile_residuals.R
```

Run it from the DRModels.jl root, with drmTMB installed and `julia` on PATH. It takes about 75 s, including the 600 bootstrap refits. The run is deterministic: a second run reproduced every number above.
