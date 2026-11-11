# Gaussian "Group F" and the wave-9 unverified shapes: which fit now, how they agree with R (2026-11-09)

Reader: whoever owns R-Julia parity of the Gaussian mean routes. Purpose: say, for each shape that `docs/dev-log/2026-11-09-gaussian-sd-gaps.md` (wave 9) left refused or unverified, whether Julia now fits it, how closely it agrees with drmTMB 0.8.1 at the optimum, and what stays refused and why. Base: DRModels.jl wave-10 tip `0bfcde2bb`.

## Shapes considered, and the decision rule

Each shape got one of three verdicts, decided from what R does, before any Julia code for it was written:

- (a) R fits and converges, and Julia needs a bounded change: implement, pre-register thresholds, compare.
- (b) R does not fit it, or does not converge on the check datasets: record, keep refused by name.
- (c) R fits it but the change is too large for this lane: record with an effort estimate.

R probe (drmTMB 0.8.1 on the committed `gsd_s1` data and, for G-shapes and F-shapes, the three `gsf` datasets below; no Julia result read for these shapes; Julia at the base tip refuses each of them, or already fits it):

| Shape | R | Julia at base | Verdict |
| --- | --- | --- | --- |
| F. mean `(1 \| p \| g)`, `sigma ~ 1 + (1 \| p \| g)`, `sd(g) ~ z` | fits, converged | refused ("sd() submodels cannot be combined with a random effect on sigma") | (a) |
| F'. same without the label (independent mean and sigma intercepts on `g`, or sigma intercept on another grouping `g2`), `sd(g) ~ z` | fits, converged | refused, same message | (a) |
| G1. block beside a slope on another grouping, `(1 + x1 \| g) + (0 + x2 \| g2)`, with or without `sd(g2, coef) ~ …` | fits | refused ("only be combined with random intercepts") | (a) |
| G2. block beside an independent slope on the same grouping, `(1 + x1 \| g) + (0 + x3 \| g)` | fits, no SD | refused, same message | (a) for no SD |
| G3. two correlated blocks on different groupings, `(1 + x1 \| g) + (1 + x2 \| g2)`, with or without `sd(·, coef)` | fits | refused, same message | (a) |
| G4. block beside intercept and slope on another grouping, `(1 + x1 \| g) + (1 \| g2) + (0 + x2 \| g2)` | fits | refused, same message | (a) |
| G5. block with `sd()` on a coefficient beside an independent slope on the same grouping | R refuses ("cannot yet combine another random-effect coefficient on the same group") | refused | (b) |
| G6. q = 3 block beside a slope on another grouping, `(1 + x1 + x2 \| g) + (0 + x3 \| g2)` | returns a log-likelihood, but `is_converged()` is FALSE on all three `gsf` datasets (and `pdHess` FALSE on one) | refused | (b): no converged reference |
| H. a label shared by two terms on one grouping, `(1 \| p \| g) + (0 + x1 \| p \| g)` | fits, but the label is a no-op: R's log-likelihood on `gsd_s1` (-291.1061) equals the independent pair `(1 \| g) + (0 + x1 \| g)` to every printed digit, and `corpars` is empty; the correlated block `(1 + x1 \| g)` gives -286.8266 | refused | (b): R silently ignores the label. Julia keeps refusing, with wording that says what R does and how to ask for either model |
| I. REML on any of the Gaussian SD-gap routes | not a gap in R | refused by name | owned by the Gaussian REML lane; unchanged |

Already fitting at the base tip and agreeing with R on `gsd_s1` (log-likelihoods equal to the printed digits): independent intercept and slope beside intercepts on other groupings, `(1 | g) + (0 + x1 | g2)` (with and without `sd(g, coef(intercept)) ~ z`), `(1 | g) + (0 + x1 | g) + (1 | g2)` with and without SD formulas, and the four-term indep shape. These need no change.

## Pre-registration (written before any Julia result for these shapes was read)

Data: `test/data/gaussian_sd_groupf/gsf_s{1,2,3}.csv` (n = 300, `g` 30 levels, `g2` 15 levels, made by `gen_data.R`; response `y` for the G shapes, `yf` for the F shapes). Reference: drmTMB 0.8.1, `engine = "tmb"`, `r_reference.R`, output `r_reference.tsv`, on Totoro (`~/drm_campaign_20261108/Rlib`).

Models: `g_blk_slope`, `g_blk_slope_sd`, `g_blk_indep`, `g_two_blk`, `g_two_blk_sd`, `g_blk_int_slope`, `g_blk_int_slope_sd` (G1 to G4), `f_corr`, `f_indep`, `f_cross` (F, F'). Refusal checks (no R comparison possible): `g_blk_indep_sd`, `g_blk_indep_sd2`, `g_q3_slope`.

Quantities compared per fit, thresholds fixed now (the wave-9 thresholds, unchanged):

| Quantity | Pass if |
| --- | --- |
| log-likelihood | absolute difference at most 1e-4 |
| fixed effects (`mu`, `sigma`) and SD-formula coefficients | maximum absolute difference in R standard-error units at most 0.01 |
| natural-scale SD of a term with no SD formula (including the sigma random-effect SD, log scale of the SD) | maximum absolute log-scale difference in R standard-error units (SE of log SD = R SE / SD) at most 0.05 |
| correlations (block, and mean-sigma `rho`) | maximum absolute difference at most 0.01 |

Boundary flag, as in wave 9: any fitted SD below 1e-3 or any |correlation| above 0.99 on either side, or R `is_converged() = FALSE` or `pdHess = FALSE`, or Julia `converged = false`. Flagged fits are listed and excluded from the pass count. A shape counts as implemented only if every non-flagged fit on all three datasets passes (and at least two datasets are non-flagged). Anything else is refused again and reported. F-group fits are Laplace in both programs (the same objective), so the same thresholds apply; a failure there opens an argument, not a threshold change.

## What was implemented

| Shape | Route | Code |
| --- | --- | --- |
| G1 to G4. correlated `(1 + x \| g)` blocks (q = 2) beside independent intercepts and slopes on any grouping, two blocks on different groupings, optional `sd(·, coef(·)) ~ …` | `_corr_crossed_split` now returns the block positions and the independent-term positions (every non-block term an intercept or a `0 + x` slope); `_fit_gaussian_corr_crossed` takes a list of blocks and a list of independent terms, each independent term carrying an optional row loading `w` (a slope is an intercept component weighted by `x`, the wave-9 device of `_LssComp`). Marginal `V = diag(σ²) + Σ_blocks U Σ U' + Σ_terms s² (w w') ∘ (Z Z')`, dense exact ML, 5000-row cap. A single block beside intercepts is bit-identical to wave 9 (same parameter order, names and keys). The second and later blocks store their correlation under `:recov_<group>` | `src/gaussian_corr_crossed_sd.jl` |
| F, F'. mean `sd(g) ~ …` with a `sigma` intercept, independent or `(1 \| p \| g)` in both | the sigma-side dense Laplace engine (`_fit_gaussian_sigma_sd_laplace`) already carried a mean SD design, a sigma SD design and a mean-sigma correlation; the router now builds the sigma SD design `~ 1` when no sigma-side `sd()` formula exists. The correlated fit stores typed SD blocks plus `:cor` (rho = tanh) instead of `:resd`, so `re_sd`/`vc` refuse by name as for every SD regression | `src/gaussian_sigma_sd_laplace.jl`, `src/gaussian_core.jl` |

Refusals added or reworded (all by name): an `sd()` formula on a grouping that carries a block and another term (R's message is quoted); two blocks on one grouping; a coefficient that is both in a block and a separate term ("overlapping"); two blocks when either has more than two coefficients and a q = 3 or 4 block beside a slope term (not checked / R does not converge); an `sd()` formula naming no term; a shared label on one grouping (now says what R does); REML on all of the above.

## Agreement with R at the optimum (30 fits: 10 models x 3 datasets)

Julia 1.10.12 on Totoro, one thread per process, three processes at once, about 5 minutes wall (estimate 10 minutes; nothing left running). Per-fit rows: `2026-11-09-gaussian-sd-groupf-compare.tsv` (regenerate with `tools/gaussian_sd_groupf_compare.jl`).

| Group | Fits | Flagged | Pass | median / max \|ΔlogLik\| | max fixed-effect gap (R SE) | max SD-formula gap (R SE) | max log-SD gap (R SE) | max \|Δcorrelation\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 to G4, block beside slopes (7 models) | 21 | 0 | 21 | 3.2e-12 / 1.0e-11 | 1.3e-9 | 2.2e-9 | 1.2e-9 | 2.3e-10 |
| F, F' mean `sd()` + sigma intercept (3 models) | 9 | 0 | 9 | 9.3e-11 / 4.7e-9 | 1.6e-8 | 2.0e-8 | 5.7e-9 | 5.8e-9 |

Thresholds were 1e-4, 0.01, 0.01, 0.05, 0.01, so every gap is at least five orders of magnitude inside them. No fit was flagged (no SD below 1e-3, no |correlation| above 0.99, R converged with a positive-definite Hessian, Julia converged), and no key was unmatched in either direction. All shapes pass on all three datasets. Agreement this close is expected: both programs optimise the same objective (exact marginal for G, Laplace value for F), so the gaps are optimiser tolerance, not model differences.

## Verdicts

| Shape | Verdict | Outcome |
| --- | --- | --- |
| F: mean `(1 \| p \| g)` + sigma `(1 \| p \| g)` + `sd(g) ~ z` | (a) | fits; 3 of 3 datasets pass |
| F': independent mean and sigma intercepts, same or different grouping, `sd(g) ~ z` | (a) | fits; 6 of 6 pass |
| G1. block beside a slope on another grouping, with and without `sd(g2, coef) ~ …` | (a) | fits; 6 of 6 pass |
| G2. block beside an independent slope on the same grouping, no SD | (a) | fits; 3 of 3 pass |
| G3. two q = 2 blocks, with and without SD formulas | (a) | fits; 6 of 6 pass |
| G4. block beside intercept and slope on another grouping, with and without SD formulas | (a) | fits; 6 of 6 pass |
| G5. `sd()` formula on a grouping carrying a block and another term | (b) | R refuses (checked with `sd` on the block coefficient and on the extra slope, all three datasets); Julia refuses with R's reason |
| G6. q = 3 or 4 block beside a slope | (b) | R returns a fit that did not converge on all three datasets; refused |
| H. label shared by two terms on one grouping | (b) | R ignores the label (log-likelihood equals the independent pair); refused, message says so |
| two blocks, either with q >= 3 | (c)-like | not compared with R (R fits two q = 2 blocks; larger pairs not tried); refused as unchecked. Estimate to close: one more reference run, no code change (the engine takes any q up to 4) |
| REML for any of these | out of lane | refused by name; belongs to the Gaussian REML lane |
| mean `sd()` with a `sigma` random SLOPE, or a mean slope, or two sigma terms | not tried | refused by name; R's behaviour not probed |

## Not verified

- Julia standard errors were not compared with R's. The reference stores R's standard errors; Julia's come from the Hessian of the same objective (dense route: ForwardDiff; F route: finite differences of the analytic gradient), but I did not diff them.
- Accessors on the new fits: a smoke run on `gsf_s1` (Julia 1.10) shows `summary`, `coeftable`, `stderror`, `vcov`, `confint`, `fitted`, `residuals`, `aic` work; `ranef` works on the F routes and refuses by name on the dense routes (no stored conditional modes); `re_sd`, `vc` and `simulate` refuse by name (SD-regression fits; default `re = :marginal`). `predict` with `newdata`, bootstrap and the R bridge were not exercised.
- Data: three simulated datasets, n = 300, 30 and 15 groups, one family of shapes per response. No near-boundary or small-group stress; no dataset has the g2 slope SD or the mean-sigma correlation near zero.
- The F routes are Laplace in both programs. Agreement is at the Laplace value, not against the exact marginal; the Laplace error itself (small groups of 10 rows, non-Gaussian marginal in `sigma`) is common to both and not measured here.
- Cost: dense fits at n = 300 took 1 to 34 s on Totoro (one thread, three jobs at once, compile included in the first). Not benchmarked beyond n = 300; the cap is 5000 rows but the route is slow well before that.

## Test results

Touched-route tests plus every test file mentioning the changed refusal wording (29 files), Julia 1.10.12 and 1.12.7 on Totoro: all pass. Two files needed edits because a refusal became a fit: `test_gaussian_sd_gaps.jl` (the block-beside-slope and mean-`sd()`-plus-sigma-intercept refusal checks) and `test_gaussian_joint_mu_sigma_sd.jl` (mean `sd(id)` beside `sigma ~ 1 + (1 | site)` now fits; the refusal check uses a sigma slope instead).

Full suite, every file once on each version (606 discovered files, 16 time-balanced shards per version plus `test_iid_sd_quadrature_accuracy.jl` run separately, wall 8 to 20 minutes per version, run at commit `7f59a007c`; a later commit changed one documentation file only):

- Julia 1.10.12: all files pass.
- Julia 1.12.7: five files fail, all of them the known pre-existing 1.12 failures: `test_stored_loglik_at_optimum.jl`, `test_binomial_spatial_sd_regression.jl`, `test_locscale_raw_k_regression.jl`, `test_relmat_raw_k_regression.jl`, `test_reml_q4_missing_response.jl`. No other failure, so none needed a rerun on the base tip.
