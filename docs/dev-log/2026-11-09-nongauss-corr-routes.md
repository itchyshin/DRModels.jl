# Non-Gaussian correlated-block cells of the wave-10 gap map: design, thresholds, verdicts

Date 2026-11-09. Base: `claude/wave10-binomial-restarts-20261109` (`0bfcde2bb`). Branch `claude/nongauss-corr-20261109`.
Reader: whoever owns the R-Julia parity lane and reviews the draft PR.

## Scope

Ten cells from `docs/dev-log/2026-11-09-gap-map-wave10.md`, all ordinary (one grouping, no structured term), ML, `sigma ~ 1` where the family has `sigma`. Cell ids are the map's.

| Rank | Cells | Where Julia refused (exact text, source) |
| --- | --- | --- |
| 1 | `binomial ordinary q3 sd-none`, `q4 sd-none` | "DRModels.jl supports `(1 \| g)`, `(0 + x \| g)`, `(1 + x \| g)` and the correlated blocks `(1 + x + z [+ w] \| g)` … fitted by maximum likelihood only" (`_re_kind`, `src/gaussian_ranef.jl`, reached from `_drm_binomial_fit` because a bare q >= 3 block never reached the SD route) |
| 21-31 | `poisson ordinary q2 sd-first`, `sd-last` | "Poisson `(1 + x1 \| g)` SD regression needs one SD formula for the intercept and one for the slope" (`_drm_count_ordinary_q2_sd`, `src/ordinary_q2_sd.jl`) |
| 21-31 | `nb2 ordinary q2 sd-first`, `sd-last` | "NegBinomial2 `sd(g)` supports `(1 \| g)` or one independent numeric slope; correlated blocks are not implemented on this route" (`src/negbinomial.jl`: the unlabelled form is not an ordinary-q2 block, and the two-field route needs two targets) |
| 21-31 | `lognormal ordinary q2 sd-one`, `sd-z` | "LogNormal SD regression currently requires one ordinary mean-effect `sd(group) ~ …` formula" (`src/lognormal.jl`, two targets) |
| 21-31 | `lognormal ordinary q2 sd-first`, `sd-last` | "LogNormal `sd(g)` supports `(1 \| g)` or one independent numeric `(0 + x \| g)` term; correlated blocks are not implemented" (`_drm_lognormal_sd_regression`) |

Not taken: Binomial `ordinary q3 sd-z` and `q4 sd-z` (non-converged and timeout; owned by the non-converged triage lane), the LogNormal `crossed q1` cells (rank 10, crossed intercept, not a correlated block), q = 3, 4 for Poisson, NB2 and LogNormal (drmTMB refuses them, so there is nothing to match).

## How Gaussian and Binomial q = 2 handle the shape

- Gaussian: `_drm_gaussian_ordinary_q34_sd` (`src/gaussian_ordinary_q34_sd.jl`) takes any subset of the block's coefficients as SD targets, gives the rest a constant SD (`ones(G, 1)` design), and fits the exact marginal. Its `theta` layout matches drmTMB.
- Binomial SD route: `_drm_binomial_sd_regression` already gives an untargeted coefficient a constant SD (`sdconst_<member>` block), so q = 3, 4 with one target (`sd-first`) fits today. Only the zero-target case was refused, one layer earlier.
- Poisson and labelled NB2: `_drm_count_ordinary_q2_sd` demanded exactly two targets.
- LogNormal: every existing SD route is "Gaussian on log y plus the Jacobian" (`_fit_lognormal_sd_regression`), so the correlated block can reuse the Gaussian route above the same way.

## Design and effort

| Cell group | Change | Effort | Risk |
| --- | --- | --- | --- |
| Poisson, NB2 `first`/`last` | `_drm_count_ordinary_q2_sd` accepts 1 or 2 targets and gives the other coefficient a constant SD (the Gaussian pattern). `_ordinary_q2_sd_block` sends the unlabelled NB2 block with exactly one target to it (two targets stay on the two-field route, unchanged). | S | The post-fit bootstrap simulator for this block is guarded on two targets; one-target fits fall through it (recorded as a limit). |
| LogNormal q = 2 | `_drm_lognormal_corr_sd`: call `_drm_gaussian_ordinary_q34_sd` on `log y`, subtract `sum(log y)`. q = 2 only. | S | Gaussian fit object reused field by field; same wrapper as the independent-slope route. |
| Binomial q = 3, 4, no SD | `_binomial_bare_corr_block` routes a bare correlated block of three or four coefficients (logit, ML, `:LA`) to the SD route with no targets; every coefficient SD is the default constant. | S to route, M to settle agreement | R's own fit at q = 4 is not converged (`pdHess = FALSE`). Julia's optimum sits above R's (see verdicts). |

## Pre-registered agreement thresholds (written before any new fit was compared)

1. **Agree**: both converge and `|logLik_Julia - logLik_R| <= 1e-4`. This is the gap map's threshold.
2. **Same objective, different optimum or integrator** (Binomial only): Julia above R by more than 1e-4. To call this "integrator" and not "a worse R optimum" the Julia fit must reproduce R's value when run with a one-node (Laplace) rule, within 1e-3. Otherwise record it as a disagreement, not as agreement.
3. **R does not converge** (`is_converged() = FALSE` or `pdHess = FALSE`): record only; leave the route refused unless Julia's fit is also converged, in which case the cell ships as "Julia ahead of an unconverged R reference" with no agreement claim.
4. **Julia below R by more than 1e-4**: try `_q2_boundary_restarts` (as in `src/binomial_sd_regression.jl`); keep only if it raises the objective by a clear margin on the cell and does not change any cell that already agreed.
5. **Not bounded** (a new likelihood or an optimiser rewrite): record as too large, leave refused.
6. No tolerance is widened to make a cell pass. Tests state the observed gap.

## Verdicts

Filled in below once the runs finish.
