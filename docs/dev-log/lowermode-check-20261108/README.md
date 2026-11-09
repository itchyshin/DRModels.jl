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
