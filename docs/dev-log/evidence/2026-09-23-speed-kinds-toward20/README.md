# DRModels speed kinds toward 20 (2026-09-23)

Purpose: bank diversity first-wave cells from
`2026-09-23-speed-report-20x3-diversity.md` §4.2, plus an earlier Mac Studio
bridge/fixture wave. Soft A3 (`sections_fine`) skipped (restore >30 min).

Active lenses: Shannon. Spawned subagents: none.
Lane: `cursor:DRM.jl-speed-kinds-toward20` · worktree
`~/local-scratch/lanes/DRM.jl-speed-kinds-20260923`.

## Protocol

| Field | Value |
|---|---|
| Tip SHA | `e9d50a110` (post-#807) |
| Julia | 1.12.6 Totoro · `JULIA_NUM_THREADS=1` · BLAS=1 |
| R / drmTMB pair | skipped on Totoro wave (host load; harness note on H2H cell) |
| Hosts | Mac Studio (bridge/fixture wave) · Totoro EPYC (diversity first-wave) |

## Wave A: Mac Studio (+8; 10 → 18)

| cell_id | Comparator | Julia med_s | TMB med_s | × | Notes |
|---|---|---:|---:|---:|---|
| `drm-bridge-student` | vs TMB H2H | 0.002791 | 0.020 | **7.2** | #803 six-cohort |
| `drm-bridge-biv-rho12` | vs TMB H2H | 0.001319 | 0.020 | **15.2** | #803 |
| `drm-bridge-gamma` | vs TMB H2H | 0.000808 | 0.016 | **19.8** | #803 plus5 |
| `drm-bridge-beta` | Julia tip abs | 0.001356 | FAIL | n/a | R arm fenced |
| `drm-missing-gauss-n1000` | shared CSV H2H | 0.001177 | 0.033 | **28.0** | Δll ≈ 2.4e-4 |
| `drm-sigma-re-G40` | shared CSV walls | 0.001696 | 0.113 | **66.6** | Δll ≈ 0.065 disclosed |
| `drm-tweedie-fe-n400` | shared CSV H2H | 0.023955 | 0.061 | **2.55** | Δll ≈ 4e-5 |
| `drm-phylo-beta-p128` | tip abs | 0.023632 | n/a | n/a | covers plan `drm-phylo-beta` |

## Wave B: Totoro diversity first-wave (+10)

Board CSV: `board_drm_first_wave_20260923_e9d50a110.csv`.
Drivers: `run_first_wave.jl` + `run_phylo_repair.jl` (Distributions fix).

| cell_id | kind | Julia med_s | logLik | Notes |
|---|---|---:|---:|---|
| `drm-phylo-poisson` | D-phylo-pois | 0.022278 | −647.7 | Julia abs p=100 |
| `drm-phylo-nb2` | D-phylo-nb2 | 0.034751 | −708.6 | Julia abs p=100 |
| `drm-phylo-binomial` | D-phylo-binom | 0.023230 | −901.0 | cbind; p=128 |
| `drm-phylo-gamma` | D-phylo-gamma-beta | 0.071680 | −440.5 | Julia-only fence OK |
| `drm-h2h-q4-vs-tmb-p1000` | D-gauss-q4-phylo | 20.814729 | NaN | Julia arm only; TMB pair not run |
| `drm-crossed-binomial` | D-crossed-family | 0.037384 | −1725.6 | small G=H=20 |
| `drm-biv-gauss-rho12` | D-biv-gauss | 0.057182 | −5332.7 | residual rho12 |
| `drm-profile-ci-locscale` | D-profile-ci | 0.015969 | −579.2 | profile CI wall |
| `drm-animal-gauss` | D-animal | 0.017705 | −272.0 | animal() A supplied |
| `drm-lss-sd-slope` | D-lss | 0.002877 | −392.0 | sd(id) ~ sex |

## Rose fence

- No README/NEWS public speed claim.
- Totoro walls are **absolute**; do not invent × vs Mac Studio or vs TMB without a paired receipt.
- `drm-h2h-q4-vs-tmb-p1000` is Julia-absolute on the Totoro receipt (TMB pair owed when harness/load allows).
- Soft A3 measure-first beta-block **not** run (`sections_fine` only on old tip).

## Board disposition

Floor **10** + Wave A **8** + Wave B **10** = **28** `has_receipt` DRM cells
(diversity aim ~20 cleared; extra are the §4.2 first-wave set plus Wave A promotions).
