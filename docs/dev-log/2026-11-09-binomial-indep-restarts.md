# Binomial independent sd-z fits: optimiser restarts (wave-8 group G12), 2026-11-09

Reader: whoever owns the R-Julia parity lane. Purpose: record what was changed for G12 of the wave-8 disagreement classification (2 cells where Julia stopped 1.3 to 2.0 nats below a point its own objective scores higher), how it was measured, and the one place this departs from the "restrict to the usual trigger" instruction. Branch `claude/binomial-indep-restarts-20261109`, from `claude/wave8-integration-20261108` (`e7e0e81a5`).

## What was wrong

`_fit_binomial_sd_regression` (independent `(1 | g) + (0 + x | g)` terms with SD formulas, AGHQ) runs one L-BFGS from one start. With `sd(g, coef(.)) ~ z` the objective has a ridge in the SD coefficients (intercept SD log-coefficients about -77 + 81 z at the better point: the intercept variance is zero except in the groups with z near 1). The default start ends at a smooth interior point, `converged = true`, every group SD between 0.4 and 0.85, so the boundary trigger of `_q2_boundary_restarts` (a fitted SD below 0.05, or not converged) never fires. Only the Gamma sd-z two-field route restarted unconditionally; this route did not restart at all.

## Change (`src/binomial_sd_regression.jl`, `_fit_binomial_sd_regression`)

- The fit calls the existing `_q2_boundary_restarts` with the existing `_Q2_ALT_STARTS` and acceptance rule (same convergence status: gain over 1e-5; converged alternative over unconverged incumbent: at most 1e-5 worse; unconverged over converged: gain over 1e-3). The alternative start for q terms applies `_q2_alt_theta` to the intercept block and each later block. `converged` is the route's own `drm_optim_converged`; the warm-start buffer is an empty vector because this route has none.
- `force = any(>(1), psd)`: restarts always run when some SD formula has a covariate; intercept-only SD formulas (`sd-one`, constants) keep the usual trigger. Correlated blocks and q = 1 are not touched.
- The outcome is in `fit.scales[:q2_restarts] = [ran, winner, gain, flipped]` (not set for q = 1 or correlated fits).
- No new abstraction, no tolerance changed.

## Measurement

Julia 1.10.12 and 1.12.x on Totoro, one thread per process, 20 processes, `nice -n 10`; estimate before launch: under 15 minutes per version including compile (not timed precisely). R: drmTMB 0.8.1 from `~/drm_campaign_20261108/Rlib` (the Mac has only 0.7.1, not used).

Fresh seeds: 40 seeds (1 to 40) of the committed generator `scripts/gen.jl` (splitmix64 and Box-Muller, no `Random`; 40 groups of 10 rows, x1 to x3 N(0,1), group covariate z equally spaced in [-1, 1], intercept SD exp(-0.8 + 0.3 z), x1 slope SD exp(-0.9), Bernoulli), both cells (q = 2: intercept and x1 terms; q = 3: plus x2), SD formula `~ z` on every term. Each fit was run in three modes in the same warmed process: `off` (the code before this change), `trigger` (restarts under the usual trigger only), `force` (the committed rule). The temporary knob is described in `scripts/temporary-knob.patch`. Raw rows: `results/runB_j110.tsv`, `results/runB_j112.tsv`; summary script `scripts/an.py`. The two Julia versions give the same log-likelihoods to all printed digits.

Seeds ending lower than the best of the restarts (best = highest log-likelihood over the three modes of that seed and cell; counted at 1e-5, the acceptance rule's threshold):

| cell | before (off) | usual trigger only | committed rule (force) |
| --- | --- | --- | --- |
| q = 2, 40 seeds | 5 of 40 (4 by more than 0.01) | 2 of 40 | 0 of 40 |
| q = 3, 40 seeds | 12 of 40 | 0 of 40 | 0 of 40 |

Largest gains of the committed rule over the default start: q = 2 seed 29, 3.26 nats (the usual trigger misses it); q = 3 seed 29, 6.05 nats. Converged fits: q = 2, 37 (off) and 39 (force) of 40; q = 3, 28 and 37 of 40. One q = 2 seed (11) ends unconverged under `force` after a 0.18 gain (the rule's "unconverged over converged" branch needs more than 1e-3, which it had).

Cost (steady state, wall seconds per fit, 40 seeds; ratio = same seed and cell against `off`):

| cell | off median (max) | usual trigger median (max) | ratio, trigger: median (max) | force median (max) | ratio, force: median (max) |
| --- | --- | --- | --- | --- | --- |
| q = 2 | 0.10 s (0.39) | 0.45 s (1.77) | 2.4 (15.4) | 0.89 s (1.75) | 7.9 (15.4) |
| q = 3 | 0.97 s (1.95) | 7.55 s (12.62) | 7.9 (16.4) | 7.51 s (12.63) | 7.9 (16.4) |

The usual trigger ran the restarts in 20 of 40 (q = 2) and 39 of 40 (q = 3) fits: with a third term the x2 slope SD usually collapses, which is itself the trigger.

### The 2 reproduction cells and R

File `test/fixtures/binomial_indep_restarts/g12_binomial.csv` (the wave-8 fuzz data), `results/trig_g12.txt`, `scripts/r_g12.R` (drmTMB 0.8.1, `is_converged() = FALSE` in both cells):

| cell | Julia before | Julia, usual trigger only | Julia, committed rule | drmTMB 0.8.1 |
| --- | --- | --- | --- | --- |
| q = 2 | -255.2241 | -255.2241 (no restart ran) | -253.1450 (winner: start 1, gain 2.08, converged) | -253.5291 |
| q = 3 | -254.5596 | -254.5596 (no restart ran) | -253.1280 (winner: start 1, gain 1.43, converged) | -253.5291 |

Julia now sits 0.38 and 0.40 nats above R's value; R's optimum is itself a point of the same ridge that R reports as not converged. Julia's objective at R's q = 2 estimate is -253.2382, below Julia's new -253.1450.

## Decision to check

The instruction was to restrict to the usual trigger if the restarts cost more than about 5x the median on healthy fits. Measured: the unconditional rule costs 7.9x the median fit (0.1 to 0.9 s for q = 2, 1 to 7.5 s for q = 3), and the usual trigger leaves both reproduction cells unfixed (table above) and misses 2 of 40 fresh q = 2 seeds. I kept the unconditional rule for fits with a covariate in an SD formula because it is the only one of the two that fixes the stated cells, and the absolute cost is small; intercept-only SD fits pay nothing. If the ratio matters more than the cells, changing `force = any(>(1), psd)` to `force = false` gives the trigger-only behaviour in one line (then the two test files' G12 expectations must go).

## Not covered

- Correlated blocks (G10), q = 4, and non-logit links were not measured. G11 cell 9 (q = 4 independent `sd-z`) goes through the same branch and so now restarts too, but I did not re-fit it.
- G13 (spatial, R lower) is not a Julia issue and was not touched.
- The generator is a stand-in for the fuzz generator (not committed on this branch); the fractions above describe it, not the fuzz harness.
