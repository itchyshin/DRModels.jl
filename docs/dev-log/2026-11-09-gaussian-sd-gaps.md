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
| C. labelled block that is the only term carrying the label | `(lhs \| p \| g)` with one use of `p` is the unlabelled block; same likelihood (R logLik of the labelled and unlabelled cells equal) | The ordinary q2 block route (`_fit_correlated_ranef_gaussian`) and the q3/q4 route; only the parser refused | Parser: unwrap the label when exactly one term carries it and `sigma` has no random effect | Low. A label shared by two terms ties correlations; refused by name, not dropped. A label on `mu` with a random effect on `sigma` is left to the existing mean-sigma route | Implement |
| D. q = 3, 4 block, no SD | Plain unstructured correlation block, TMB `UNSTRUCTURED_CORR_t` | `_fit_gaussian_ordinary_q34_sd` already evaluates exactly this per group with Woodbury; with no `sd()` target every coefficient takes a constant SD | Route the SD-free single block to the existing engine | Low. Same code that the SD cases use | Implement |
| E. crossed, no SD (q = 2, 3, 4), and B's engine | block on `g` plus an independent intercept on `g2`, dense or sparse marginal | Dense multi-component engine (`_fit_gaussian_lss_multi`) has iid and phylo components only; q34 engine is single-block per group | New dense engine: `V = diag(σ²) + Σ_g U_g Σ_g U_g' + Σ_h s_h² Z_h Z_h'`, ForwardDiff, 5000-row cap; reuses `_q34_corr_chol` so `theta` is TMB's | Medium. Dense n×n, about 30 to 50 s at n = 400 on a Mac; no new estimator | Implement |
| B. crossed with SD (q = 2) | `sd(g, coef)` per block coefficient and `sd(g2)` | Same new dense engine, with per-coefficient and per-intercept group-level SD designs (`_sd_group_design`) | Same engine, SD designs passed in | Medium | Implement |
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

Deviation from the pre-registration text: the first R reference extraction missed the correlation of a labelled q = 2 block (R reports it under `corpars$mu`, not `corpars$re_cov`). I fixed `r_reference.R` to read every `corpars` entry and re-ran the reference. No threshold changed, no Julia result had been read.

## What was implemented

| Group | Route | Code |
| --- | --- | --- |
| C. labelled block | `_split_ranef(…; unwrap_label = true)` (mean formula only, and only when `sigma` has no `(… \| g)` term) turns `(lhs \| p \| g)` into `(lhs \| g)` when `p` labels exactly one term; a label shared by two terms is refused ("shared by N") | `src/gaussian_ranef.jl`, `src/gaussian_core.jl` |
| D. q = 3, 4 block, no SD | single block, ML, no `sigma` random effect: `_drm_gaussian_ordinary_q34_sd` with no SD targets | `src/gaussian_core.jl` |
| E, B. block beside intercepts | new dense engine `_fit_gaussian_corr_crossed`; optional `sd(g, coef(·))` per block coefficient and `sd(h)` per extra intercept; ML only; 5000-row cap | `src/gaussian_corr_crossed_sd.jl` |
| A. indep with SD | `_LssComp` gains a per-row loading `w` (an independent slope is an intercept component weighted by `x`); `_drm_gaussian_indep_sd` matches `sd(g, coef(·))` targets to terms and calls `_fit_gaussian_lss_multi`; ML only | `src/gaussian_lss.jl`, `src/gaussian_corr_crossed_sd.jl` |

One consequence beyond the 23 gap cells: a q = 3 or 4 block with `sd(g, coef(·))` formulas beside a `(1 | h)` term now fits (it used to be refused). R fits it as well; one check, `gsd_s1`, `(1 + x1 + x2 | g) + (1 | g2)` with `sd(g, coef = "x1") ~ z`: R and Julia logLik both -260.418857 (difference 1e-8).

## Agreement with R at the optimum (45 fits: 15 models x 3 datasets)

Julia 1.10.12 on Totoro, one thread per process, 20 processes at once, about 4 minutes wall (estimate was 10 minutes; nothing left running). Per-fit rows: `2026-11-09-gaussian-sd-gaps-compare.tsv`. Medians and maxima are over non-flagged fits.

| Group | Fits | Flagged | Pass | median / max |ΔlogLik| | max fixed-effect gap (R SE) | max SD-formula gap (R SE) | max log-SD gap (R SE) | max |Δcorrelation| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A indep, SD formulas | 15 | 0 | 15 | 2.4e-12 / 7.1e-12 | 5.3e-10 | 9.3e-10 | 1.3e-9 | n/a |
| B crossed, SD formulas | 9 | 0 | 9 | 5.5e-12 / 6.0e-12 | 8.4e-10 | 1.6e-9 | 5.2e-10 | 3.1e-10 |
| C, D ordinary (labelled q2, q3, labelled q4) | 9 | 1 | 8 | 3.8e-12 / 5.9e-12 | 2.0e-9 | n/a | 1.8e-9 | 8.4e-10 |
| E crossed, no SD | 12 | 0 | 12 | 3.8e-12 / 7.9e-12 | 1.1e-9 | n/a | 1.8e-9 | 8.1e-10 |

Thresholds were 1e-4, 0.01, 0.01, 0.05, 0.01, so every gap is at least seven orders of magnitude inside them. Boundary flags: one fit, `gsd_s3` `ord_q4_lab`, where R reports `is_converged() = FALSE` and `pdHess = FALSE` (non-finite SEs); its logLik difference is 3.3e-5, inside the threshold, and it is excluded from the pass count as pre-registered. No fit had an SD below 1e-3 or a correlation above 0.99 in either program. All four groups pass on all three datasets.

The same routes on the original fuzz grid: all 23 Gaussian ordinary, indep and crossed ML cells that wave-8 Julia refused and R fits now fit, converged, with logLik within 5.9e-11 of R's (`2026-11-09-gaussian-sd-gaps-fuzz/results.tsv`, 13 to 33 s per cell on Totoro). The grid had 24 such rows in the gap map listing; the 24th is group F, below.

## Refused, with reasons

| Cell | What now happens | Why it stays refused |
| --- | --- | --- |
| F. `gaussian|ordinary|lab0|q1|sd-z|cm1|ML`: mean `(1 \| p \| g)` with `sigma ~ 1 + (1 \| p \| g)` and `sd(g) ~ z` | `sd() submodels cannot be combined with a random effect on sigma. R drmTMB fits … ` | A mean SD formula together with the mean-sigma correlated intercept pair is a new marginal (two correlated effects, one with a covariate-dependent SD), not a reuse of the dense or Woodbury engines; the sigma-side Laplace route covers `sd(g, sigma)` only. Stop at this note |
| REML on any new shape | the generic REML gate ("method = :REML is not implemented …") or, on the SD routes, "supports maximum likelihood only" | Owned by the Gaussian REML lane; this branch does not touch the REML gate, so none of these routes can relabel an ML fit as REML |
| a block beside a slope term, `(1 + x \| g) + (0 + z \| h)` | "a correlated `(1 + x \| g)` block can only be combined with random intercepts" | Not in the gap map; no R comparison made |
| crossed q = 3, 4 with `sd()` on the other grouping | not a gap | R refuses it ("can only target that block") |
| a label shared by two terms | "the label `p` … is shared by N random-effect terms" | It ties correlations between terms; no R comparison made |

## Not verified

- Dense route cost: about 30 to 45 s per crossed fit at n = 240 to 400 on Totoro (one thread, 20 jobs at once). Not benchmarked beyond n = 400; the cap is 5000 rows but the route will be slow well before that.
- Standard errors were compared in R SE units only through the point estimates. Julia SEs themselves were not compared to R's (the reference stores R SEs; the Julia Hessian is the same dense objective, but I did not check it).
- Predictions, residuals, `ranef`, simulation and the R bridge on the new routes: not wired or tested. The corr-crossed route attaches no conditional modes and the indep route none beyond what `_fit_gaussian_lss_multi` already gives; I did not call `ranef` on either.
- Only one dataset family: three simulated datasets of 240 rows with 30 and 12 groups, plus the one fuzz dataset per shape. No near-boundary or small-group stress.
- Julia 1.12 not run on these fits; the tests use the committed data and R reference only and no Julia-version-specific output.
- Tests: `test_gaussian_sd_gaps.jl` (12 fits and 8 refusal checks, passes on Totoro in about 5 minutes), plus 15 related existing test files that pass; two of them (`test_implicit_re_intercept.jl`, `test_gaussian_ordinary_q3_q4_sd.jl`) asserted old refusals that are now fits and were updated. The full Julia suite was not run.
