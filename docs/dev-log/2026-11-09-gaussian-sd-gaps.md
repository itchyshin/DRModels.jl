# Gaussian ordinary and crossed SD gaps: ranking, routes, agreement with R (2026-11-09)

Reader: whoever owns R-Julia parity of the Gaussian mean routes. Purpose: say which refusals in the wave-8 gap map (`drmTMB` branch `claude/gap-map-wave8-20261108`, `docs/dev-log/parity-crosswalk-evidence-20261107/fuzz-gap-map-wave8/`) this branch closes, how each was closed, how well it agrees with R drmTMB 0.8.1 at the optimum, and what stays refused and why.

## The cells

All are Gaussian, ML, `sigma ~ 1`, 400 rows in the fuzz grid. R fits every one; wave-8 Julia refused them. The ranking in the gap map counts cells that share one refusal message.

| Group | Cells | Model | Julia refusal at wave 8 |
| --- | --- | --- | --- |
| A. `indep` with SD formulas | 8 | `(1 \| g) + (0 + x1 \| g) [+ …]`, `sd(g, coef(·)) ~ z` (q = 2, 3, 4; first, last, one, z) | `sd() supports random INTERCEPT terms only` |
| B. `crossed` with SD formulas | 4 | `(1 + x1 \| g) + (1 \| g2)`, `sd(g, coef(·)) ~ …`, `sd(g2) ~ …` (q = 2) | same message |
| C. labelled `(.. \| p \| g)` | 10 | q = 2, 3, 4 on `ordinary` and `crossed` | `unsupported random-effect term` |
| D. q = 3, 4 correlated block, no SD | 8 (6 Gaussian) | `ordinary` and `crossed` | `supports (1 \| g), (0 + x \| g), (1 + x \| g)` |
| E. q = 2 block beside `(1 \| g2)`, no SD | 1 | `crossed`, unlabelled | `a correlated block must be the only random-effect term` |
| F. mean RE + sigma RE + `sd()` | 1 | `(1 \| p \| g)` on mu and sigma, `sd(g) ~ z` | `sd() submodels cannot be combined with a random effect on sigma` |

R limits worth knowing before reading the ranking: R refuses an `sd()` formula on another grouping when the correlated block has three or four coefficients ("can only target that block"), so crossed q = 3, 4 with SD formulas is not a gap, it is a refusal on both sides.

## What R fits, what Julia machinery exists, cost and risk

| Group | R model (drmTMB `R/`, `src/drmTMB.cpp`) | Reusable Julia machinery | Cost | Risk | Decision |
| --- | --- | --- | --- | --- | --- |
| C. label on a block that is the only term carrying the label | `(lhs \| p \| g)` with one use of `p` is the unlabelled block; same likelihood (R logLik of the labelled and unlabelled cells equal) | The ordinary q2 block route (`_fit_correlated_ranef_gaussian`) and the q3/q4 route; only the parser refused | Parser: unwrap the label when exactly one term carries it and `sigma` has no random effect | Low. A label shared by two terms ties correlations; refused by name, not dropped. A label on `mu` with a random effect on `sigma` is left to the existing mean-sigma route | Implement |
| D. q = 3, 4 block, no SD | Plain unstructured correlation block, TMB `UNSTRUCTURED_CORR_t` | `_fit_gaussian_ordinary_q34_sd` already evaluates exactly this per group with Woodbury; with no `sd()` target every coefficient takes a constant SD | Route the SD-free single block to the existing engine | Low. Same code that the SD cases use | Implement |
| E, crossed no-SD (q = 2, 3, 4) | block on `g` plus an independent intercept on `g2`, dense or sparse marginal | Dense multi-component engine (`_fit_gaussian_lss_multi`) has iid and phylo components only; q34 engine is single-block per group | New dense engine: `V = diag(σ²) + Σ_g U_g Σ_g U_g' + Σ_h s_h² Z_h Z_h'`, ForwardDiff, 5000-row cap; reuses `_q34_corr_chol` so `theta` is TMB's | Medium. Dense n×n, about 30 to 50 s at n = 400 on a Mac; no new estimator | Implement |
| B, crossed with SD (q = 2) | `sd(g, coef)` per block coefficient and `sd(g2)` | Same new dense engine, with per-coefficient and per-intercept group-level SD designs (`_sd_group_design`) | Same engine, SD designs passed in | Medium | Implement |
| A, indep with SD | one independent component per coefficient, each `sd(g, coef) ~ …` | Dense multi-component engine; add a per-row slope loading `w` to `_LssComp` (an independent slope is an intercept component with weights `x`) | Small: one struct field and one factor in the covariance assembly, plus target matching | Low to medium: shared engine edited; existing components have `w = nothing` and are bit-identical | Implement |
| F | mean `(1 \| p \| g)` correlated with a sigma `(1 \| p \| g)` plus `sd(g) ~ z` | Sigma-side Laplace route (`gaussian_sigma_sd_laplace.jl`) takes `sd(g, sigma)`; no route has the mean SD formula and the mean-sigma correlation together | A new marginal (two correlated effects with a covariate-dependent SD), not a reuse | High | Refuse with accurate wording; stop at this note |

No group needs a new estimation algorithm. Groups E and B add one dense exact-marginal engine of the same kind as the existing dense engines.

## Pre-registration (written before any Julia result on the new datasets was read)

Informal smoke runs on the fuzz data (before this section was written): 13 cells fit in Julia with logLik within 1e-6 of R's reference values in the gap map (ordinary lab1 q2, q3, q4; crossed q2 none and `z`; indep q2 first, `z`, q3 `z`). That is the only result seen. The comparison below is on three new datasets and 15 models.

Data: `test/data/gaussian_sd_gaps/gsd_s{1,2,3}.csv` (n = 240, 30 groups `g`, 12 groups `g2`, simulated by `gen_data.R` from a correlated 4-coefficient block with SD depending on `z`, plus a `g2` intercept with SD depending on `z2`). Reference: drmTMB 0.8.1, `engine = "tmb"`, `r_reference.R`, output `r_reference.tsv`.

Models (15): `ord_q2_lab`, `ord_q3`, `ord_q4_lab`; `crs_q2`, `crs_q2_lab`, `crs_q3`, `crs_q4_lab`, `crs_q2_first`, `crs_q2_last`, `crs_q2_z`; `ind_q2_first`, `ind_q2_last`, `ind_q2_z`, `ind_q2_one`, `ind_q3_z`.

Quantities compared per fit, thresholds fixed now:

| Quantity | Pass if |
| --- | --- |
| log-likelihood | absolute difference at most 1e-4 |
| fixed effects (`mu`, `sigma`) and SD-formula coefficients | maximum absolute difference in R standard-error units at most 0.01 |
| natural-scale SD of a coefficient with no SD formula | maximum absolute log-scale difference in R standard-error units (SE of log SD = R SE / SD) at most 0.05 |
| correlations | maximum absolute difference at most 0.01 |

Reported per group over all non-flagged fits: median, maximum. Boundary flag: any fitted SD below 1e-3 or any |correlation| above 0.99 on either side, or R `is_converged() = FALSE` or `pdHess = FALSE`, or Julia `converged = false`. Flagged fits are listed and excluded from the pass count. A group counts as implemented only if every non-flagged fit on at least three datasets passes. Anything else is refused again and reported.
