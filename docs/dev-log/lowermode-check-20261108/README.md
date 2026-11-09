# Lower-mode check: Gamma q=1 and NB2 q=2 sd-z routes (2026-11-08)

Question: after the Gamma sd-z two-field fix (always run the shared alternative
starts), do the single-field Gamma route `sd(g, phylo|spatial) ~ z` and the NB2
two-field sd-z route also stop at a lower optimum than R?

Answer: no. In 96 fits (4 routes x 24 seeds), Julia was never more than 5.6e-7
below R's log-likelihood. No restart rule was added and no source file changed.

## Design, fixed before any result was read

- Routes: Gamma `phylo(1 | g)` and `spatial(1 | g)` with `sd(g, ...) ~ z` (q=1);
  NB2 `phylo(1 + x1 | g)` and `spatial(1 + x1 | g)` with `~ z` on both SDs (q=2).
- Data: the wave-9 generator (`gen_data.jl`; G=40 groups, 10 rows each, same seed
  formula, so seeds 1 to 10 are the wave-9 data up to float printing), seeds 1 to 24.
- R: drmTMB 0.8.1, ML, one thread. Julia: DRM.jl 479bbf963, Julia 1.10.12.
- Lower mode: `llR - llJ > 1e-3`. Higher: `llJ - llR > 1e-3`. Agree otherwise.

## Result (`per_seed.tsv` has every seed)

| Route | n | Julia lower | Julia higher | agree | min dLL (J-R) | max dLL |
|---|---|---|---|---|---|---|
| Gamma phylo q1 | 24 | 0 | 0 | 24 | -4.0e-9 | 7e-10 |
| Gamma spatial q1 | 24 | 0 | 1 | 23 | -3.4e-9 | 2.55 |
| NB2 phylo q2 | 24 | 0 | 1 | 23 | -1.0e-9 | 0.108 |
| NB2 spatial q2 | 24 | 0 | 4 | 20 | -5.6e-7 | 0.289 |

Julia higher (R sits at the lower optimum): Gamma spatial seed 20 (+2.548);
NB2 phylo seed 15 (+0.108); NB2 spatial seeds 1, 5, 6, 20 (+0.093, +0.165, +0.289,
+0.112). In NB2 spatial seed 5 Julia's own `converged` flag is false (dLL +0.165,
R also flags `is_converged = FALSE` and `pdHess = FALSE`); every other Julia fit
reports `converged = true`. R flagged `is_converged = FALSE` on 3 NB2 phylo and 6
NB2 spatial seeds. I did not diagnose these: the direction is the opposite of the
question, and nothing in the repo depends on them.

Because there is no lower-mode seed, the "evaluate R's theta in Julia's objective"
step had nothing to confirm and was not run.

## Cost

No change, so no added cost. The existing fits took at most 32 s (Gamma phylo q1),
4.5 s (Gamma spatial q1), 15.5 s (NB2 phylo q2), 11.8 s (NB2 spatial q2) on Totoro
with one thread per fit.

## What this does not cover

- Only one data-generating shape (sd-z with exp(0.3 z) scaling, SD about 0.3 to 0.5,
  40 groups). A harder design could still produce a lower mode.
- Gamma q=1 sd-z with more than one SD covariate, `lab1` correlated fields, relmat
  and animal kernels, and Gaussian-copula or other families were not run.
- The NB2 route records no restart outcome in `fit.scales`, so I cannot say how
  often its boundary-only restarts ran.
- R side lesson (not DRM.jl): drmTMB workers need `OMP_NUM_THREADS=1` on Totoro.
  Without it each R process opened 385 threads and the host load average hit about
  3000 before I stopped them.

Reproduce: generate data with `gen_data.jl`, run `agree.jl` and `agree.R` on a jobs
file with columns `route<TAB>fam|kind|lab0|qN|sd-z|cm0|ML<TAB>seed`.

## Follow-up: do the forced Gamma restarts need to vary the first field's SD z-coefficients?

A review of PR #1270 said `_q2_alt_theta` only makes the second (slope) field's SD
z-coefficients vary (`het`), so a lower mode that differs only in the first field's
SD slopes cannot be reached. Reading the code confirms the gap: the first block is
always set to a constant SD by least squares (`Zs[1] \ fill(log(sd0), ...)`).

I tested a fix (`candidate_field1_starts.patch`, not applied): `_q2_alt_theta` gets a
`het0` argument, and the forced path (`force = true`, Gamma only) runs four extra
starts `(sd0, sd1, het, het0) = (0.4, 0.22, 0, -1), (0.4, 0.22, 0, 1),
(0.4, 0.22, -1, 1), (0.4, 0.22, 1, -1)`, accepted by the same clear-gain rule.

Design, fixed before running: Gamma `phylo(1 + x1 | g)` and `spatial(1 + x1 | g)`,
`sd ~ z` on both fields, seeds 0 to 24 (the 22-seed wave-9 set is seeds 0 to 10 of
these two routes) plus fresh seeds 25 to 74; drmTMB 0.8.1 ML as reference; a seed
counts as fixed if the after log-likelihood is more than 1e-3 above before.

| Seeds | Route | Before lower than R | After lower than R | Seeds changed by extra starts | Cost after/before (median, max) |
|---|---|---|---|---|---|
| 0-24 | Gamma phylo q2 | 0 | 0 | 0 | 1.49, 1.87 |
| 0-24 | Gamma spatial q2 | 0 | 0 | 0 | 1.52, 1.74 |
| 25-74 | Gamma phylo q2 | 0 | 0 | 0 | 1.58, 1.77 |
| 25-74 | Gamma spatial q2 | 0 | 0 | 0 | 1.59, 1.81 |

At the current tip the forced restarts already put all 150 fits at or above R's
log-likelihood (R is the lower one in 9), and the extra starts change no fit while
adding about 50% to fit time. I did not ship the change: it costs time and no seed
needs it. The gap is real in principle; if a seed ever shows a first-field-only lower
mode, the patch is the candidate fix and that seed its failing-before test.

Per-seed values: `gamma_q2_field1_seeds0-24.tsv`, `gamma_q2_field1_seeds25-74.tsv`.
Not covered: 40 groups and the exp(0.3 z) SD design only; no other family.
