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
