# Aug-24 H2H re-anchor (Totoro, 2026-09-23)

**Purpose:** re-measure the soft-owed Aug-24 locscale / relmat cells on tip
`DRModels` (`1b8e81c`) vs installed tip `drmTMB` **0.7.1**, with **byte-identical
fixtures** shared across engines (CSV under `fixtures/`).

**Not** a JuliaCall `engine="julia"` bridge run. JuliaCall setup hung under
Totoro concurrent load; arms were timed separately on the same fixture bytes.
`drm_julia_load_module` on tip drmTMB already accepts `DRModels|DRM` (no
Package-DRM rename owed for this receipt).

## Machine / versions

- Host: Totoro (EPYC). Load during TMB arm ~114 (1/5/15 ≈ 114/126/112). **TMB
  walls may be load-inflated**; do not headline a quiet-machine × from the
  relmat TMB median alone.
- Threads: Julia `JULIA_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`.
- drmTMB 0.7.1 (`~/Rlibs/drmtmb-h2h-20260923`); DRModels tip `1b8e81c`.

## Method

1. R builds fixtures (locscale seed `20260815`; relmat = `engine_speed_grid.R`
   `mk_relmat(77)`), times `engine="tmb"` (1 warm + 3 timed; median).
2. Julia reads the same CSVs and times `drm(...; data=..., K=...)` the same way.
3. Correctness gate: `|Δ loglik|` negligible.

## Results (medians)

| cell_id | tmb (s) | julia (s) | tmb/julia | \|Δ loglik\| |
|---|---:|---:|---:|---:|
| `gauss_locscale_n1000` | 0.024 | 0.000553 | **43.4×** | ~0 (ll −1071.6134544765) |
| `gaussian_relmat_G25` | 8.847† | 0.000906 | n/a as quiet ×† | ~0 (ll −84.35662192746) |

† Relmat TMB median is **not** comparable to the Aug-24 quiet-host 0.110 s under
this load; Julia absolute + loglik identity are the load-bearing claims for that
cell. Locscale TMB (24 ms) is in the same ballpark as Aug-24 (25 ms) despite load.

Raw: `soft_owed_h2h_combined.tsv`, `tmb_soft_owed.tsv`, `julia_soft_owed.tsv`.

## Board disposition

- `drm-gauss-locscale-n1000`: tip H2H `has_receipt` (vs TMB + tip abs).
- `drm-gauss-relmat-G25`: tip abs + loglik identity `has_receipt`; TMB wall
  load-fenced (do not invent quiet ×).
