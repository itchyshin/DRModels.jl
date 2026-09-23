# DRModels speed kinds toward 20 (2026-09-23)

Purpose: add 8 new board kinds not in the prior 10-cell DRM matrix
(after #803–#807), moving the package count **10 → 18** toward a 20-kind target.

Active lenses: Shannon. Spawned subagents: none.
Lane: `cursor:DRM.jl-speed-kinds-toward20` · worktree
`~/local-scratch/lanes/DRM.jl-speed-kinds-20260923`.

## Protocol

| Field | Value |
|---|---|
| Tip SHA | `e9d50a110` (post-#807) for new fixture H2H; bridge promotions cite banked `cf058168b` |
| Julia | 1.10.0 · `JULIA_NUM_THREADS=1` · `OPENBLAS_NUM_THREADS=1` |
| R | 4.6 · drmTMB **0.7.1** · OMP/OPENBLAS/MKL=1 |
| Host | Mac Studio `w-kw3k3y6229.psych.ualberta.ca` (arm64) |
| Totoro | skipped this slice (load ~231); light cells only, no heavy local Julia |

## New cells (8)

| cell_id | Comparator | Julia med_s | TMB med_s | × | Notes |
|---|---|---:|---:|---:|---|
| `drm-bridge-student` | vs TMB H2H | 0.002791 | 0.020 | **7.2** | promoted from #803 six-cohort |
| `drm-bridge-biv-rho12` | vs TMB H2H | 0.001319 | 0.020 | **15.2** | promoted from #803 |
| `drm-bridge-gamma` | vs TMB H2H | 0.000808 | 0.016 | **19.8** | promoted from #803 plus5 |
| `drm-bridge-beta` | Julia tip abs | 0.001356 | FAIL | n/a | R `argument "a" is missing` fenced |
| `drm-missing-gauss-n1000` | shared CSV H2H | 0.001177 | 0.033 | **28.0** | Δll ≈ 2.4e-4 |
| `drm-sigma-re-G40` | shared CSV walls | 0.001696 | 0.113 | **66.6** | Δll ≈ 0.065 disclosed |
| `drm-tweedie-fe-n400` | shared CSV H2H | 0.023955 | 0.061 | **2.55** | Δll ≈ 4e-5 |
| `drm-phylo-beta-p128` | tip abs | 0.023632 | n/a | n/a | Beta phylo p=128; no TMB pair |

Machine-readable: `board_new_cells.tsv`. Fixtures under `fixtures/` for the three
new H2H cells.

## Rose fence

- No README/NEWS public speed claim.
- Do not mix bridge-cohort SHA (`cf058168b`) with tip fixture SHA without
  labeling the column.
- `drm-sigma-re-G40` walls are load-bearing; loglik identity is **not** claimed
  (|Δll|≈0.065).
- `drm-bridge-beta` is Julia-absolute only until the R arm is repaired.
- Totoro re-time optional; not required for this bank.

## Board disposition

Prior DRM board **10/10** `has_receipt`. After this bank: **18/18** on these
kinds (still short of 20; two slots remain for a later slice).
