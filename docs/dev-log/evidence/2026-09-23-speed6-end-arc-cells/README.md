# Speed6 end-of-arc cell walls (2026-09-23)

Retained artifact for Shinichi's end-of-arc bar: attested Julia vs drmTMB
warm medians on ≥5 benchmark cells (Gaussian + non-Gaussian + structured),
plus a q=4 phylo ML fit-wall receipt on the merged #781 tip.

Perspectives: Shannon (Cursor Grok lane). No nested subagents.

## Protocol

| Field | Value |
|---|---|
| Julia tip | `cf058168b` (merge of #781 into `main`) |
| Julia | 1.10.0 · `julia_threads=1` · `blas_threads=1` for bridge arms |
| R | 4.6.0 · drmTMB **0.7.1** · `OMP/OPENBLAS/MKL=1` |
| Bridge scripts | `bench/bridge_six_cell_timing.jl` + `bench/R/bridge_six_cell_timing.R` |
| Cohorts | #372 six + #389 plus5 (same cell ids as prior evidence; **re-timed today**) |
| Reps | 1 warmup discarded + 5 timed; report median |
| Machine | `w-kw3k3y6229.psych.ualberta.ca` (arm64 Darwin, Mac Studio) |
| q4 sections | `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1` · `--gate tsv --p 100,1000` |

Out of scope for this receipt: a universal Nx headline; a Totoro re-run of
these walls; a paired drmTMB arm for the q4 phylo synthetic grid; any claim
that speed6's cholesky reuse or warm-u0 moved the q4 fit wall (see identity
note below).

Machine-readable twins in this directory:

- `julia_bridge_six_cell.toml` / `r_bridge_six_cell.json`
- `julia_bridge_plus5.toml` / `r_bridge_plus5.json`
- `q4_sections_cf058168b.tsv`

## Paired bridge cells (10 OK; 1 R fail)

Speedup = R median / Julia median (>1 means DRModels faster).

| Cell | Class | Julia median_s | R median_s | R/J |
|---|---|---:|---:|---:|
| gaussian-locscale | Gaussian | 0.000446 | 0.022 | **49.3×** |
| gaussian-bivariate-rho12 | Gaussian + structured (rho12) | 0.001319 | 0.020 | **15.2×** |
| meta-analysis-V | Gaussian + structured (`meta_V`) | 0.001032 | 0.016 | **15.5×** |
| `robust-student` | Non-Gaussian | 0.002791 | 0.020 | **7.2×** |
| count-nbinom2 | Non-Gaussian | 0.001149 | 0.021 | **18.3×** |
| count-poisson | Non-Gaussian | 0.000235 | 0.013 | **55.2×** |
| positive-gamma | Non-Gaussian | 0.000808 | 0.016 | **19.8×** |
| binomial-trials | Non-Gaussian | 0.001942 | 0.014 | **7.2×** |
| positive-lognormal | Non-Gaussian | 0.000811 | 0.016 | **19.7×** |
| nbinom2-dispersion | Non-Gaussian | 0.001407 | 0.025 | **17.8×** |
| proportion-beta | Non-Gaussian | 0.001356 | FAIL | R arm: `argument "a" is missing` |

Across the 10 paired OK cells the median ratio is 18.3× (range 7.2× to 55.2×).

Coverage vs the end-of-arc ask: Gaussian (3), non-Gaussian (7), structured
(bivariate rho12 + meta_V; q4 phylo wall below). Aim-10 met on paired walls.

## q=4 phylo ML fit wall on tip (structured Gaussian)

From `q4_sections_cf058168b.tsv` (Julia only; chol reuse engaging, 0 fallbacks):

| p | fit_wall_s | chol_factorizations | chol_fallbacks | loglik |
|---:|---:|---:|---:|---:|
| 100 | 0.976 | 247 | 0 | (section TSV) |
| 1000 | 10.375 | 289 | 0 | −8661.6245 |

### Honest speed6 before/after (Julia tip vs banked S3)

Banked S3 partition (`q4_sections_a734d2b90.tsv`, pre-S5e reuse engaging) vs
today's tip. Same synthetic q4 route; not a paired R comparison.

| p | before wall_s (a734d2b90) | tip wall_s (cf058168b) | ratio tip/before |
|---:|---:|---:|---:|
| 100 | 0.874 | 0.976 | 1.12 (no gain) |
| 1000 | 8.331 | 10.375 | 1.25 (no gain) |

Rose reading: speed6 landed identity / hygiene (chol pattern reuse with
zero fallbacks; warm u0 into `_q4_fd_vcov`). It does not buy a fit-wall
speedup on this q4 grid. The attested end-of-arc speedups are the 10 Julia
vs drmTMB 0.7.1 bridge cells above, re-measured on the merged tip.

## Commands (repro)

```sh
git checkout cf058168b
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DRM_372_REPS=5 DRM_BRIDGE_TIMING_COHORT=six \
  julia --project=. bench/bridge_six_cell_timing.jl
DRM_372_REPS=5 DRM_BRIDGE_TIMING_COHORT=six \
  Rscript --vanilla bench/R/bridge_six_cell_timing.R
DRM_372_REPS=5 DRM_BRIDGE_TIMING_COHORT=plus5 \
  julia --project=. bench/bridge_six_cell_timing.jl
DRM_372_REPS=5 DRM_BRIDGE_TIMING_COHORT=plus5 \
  Rscript --vanilla bench/R/bridge_six_cell_timing.R
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  julia --project=. bench/profile_q4_sections.jl --gate tsv --p 100,1000
```
