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

R side rerun on Totoro (drmTMB 0.8.1, library `~/drm_campaign_20261108/Rlib`, one core, `nice`): every logLik reproduces the wave-10 map to 1e-9 (`test/fixtures/nongauss_corr/r_reference_drmTMB_0.8.1.csv`, script `run_r_nongauss_corr.R`). Julia: Julia 1.10.12, default settings, `data` as in the map.

| Cell | R logLik | Julia logLik | Julia minus R | Verdict |
| --- | --- | --- | --- | --- |
| poisson ordinary q2 `sd-first` | -712.3057715513 | -712.3057715514 | -1e-10 | (a) fits, agrees |
| poisson ordinary q2 `sd-last` | -711.3985694172 | -711.3985694171 | 1e-10 | (a) |
| nb2 ordinary q2 `sd-first` | -696.3014043646 | -696.3014043646 | 0 | (a) |
| nb2 ordinary q2 `sd-last` | -698.1310518253 | -698.1310518254 | -1e-10 | (a) |
| lognormal ordinary q2 `sd-one` | -550.8682164308 | -550.8682164308 | 0 | (a) |
| lognormal ordinary q2 `sd-z` | -549.4290610144 | -549.4290610144 | 0 | (a) |
| lognormal ordinary q2 `sd-first` | -549.9083396264 | -549.9083396264 | 0 | (a) |
| lognormal ordinary q2 `sd-last` | -550.4577955853 | -550.4577955853 | 0 | (a) |
| binomial ordinary q3 `sd-none` | -253.5747241067 | -253.00244 | +0.572 | (a) by rule 2: Julia with `nq = 1` (Laplace) gives -253.57472, equal to R to 1e-5; the default 5-node rule and `nq = 9` (-253.00057) move toward -253.0. Integrator, not a worse optimum. Ships with the gap documented, not as agreement at 1e-4 |
| binomial ordinary q4 `sd-none` | -253.3593431487 (not converged, `pdHess` FALSE) | -252.71318 (converged) | +0.646 | (b) R does not converge. Julia's default fit converges; recorded with no agreement claim. `nq = 1` at q = 4 stops at -239.34 (not converged), so the Laplace rule is not a usable comparison there |

Cells (c) and (d): none. No cell needed boundary restarts (Julia never stopped below R) and none was too large.

Julia-version note: the q = 3 Binomial fit reports `converged = true` on 1.10.12 and `false` on 1.12.6, and its logLik differs between the `~ 1`-explicit and bare forms by 1e-5 on 1.12.6 (at most 1e-6 on 1.10.12). The optimum has a flat direction (`rcond` about 1e-11, a pseudo-inverse warning). Tests do not assert the flag there and use 1e-4 for that comparison; this is evidence-based (observed gaps), not a widening to pass.

## Behaviour changes and limits

- Poisson labelled `(1 + x | p | g)` with one SD formula was refused and now fits (the labelled form is a Julia extension; drmTMB refuses it). `test/test_labelled_ordinary_q2_sd.jl` asserted the refusal and now asserts that the one-formula fit equals the explicit `~ 1` fit. That is Julia against Julia through the same new code path (there is no R reference for the labelled form, since drmTMB refuses it), so it does not check the constant-SD mapping against R; the testset keeps its old name, 'refusals stay explicit', although it now holds this acceptance check.
- `simulate` and the parametric bootstrap are refused (clear message) for the one-formula Poisson/NB2 fits, because the post-fit marginal simulator is guarded on two SD targets. `predict` and `summary` work. Not extended here.
- LogNormal q = 3, 4 and Poisson/NB2 q = 3, 4 stay refused, as in drmTMB.
- Binomial bare q = 3, 4 is logit only; other links keep the old refusal.
- Not run: R estimates and standard errors were not compared, only logLik and convergence flags.
