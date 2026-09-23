# DRM speed-board kinds addendum (2026-09-23)

Companion to the three-package speed board. This file banks the **+8** DRM
kinds that move the package from **10 → 18** `has_receipt` cells.

Authoritative numbers: `docs/dev-log/evidence/2026-09-23-speed-kinds-toward20/`.

## Added rows (append to §2.2)

| package | cell_id | DGP kind | script / fixture | baseline | current | wall before | wall after | speedup | notes | status |
|---|---|---|---|---|---|---|---|---|---|---|
| DRModels | `drm-bridge-student` | Student-t FE n=180 | `bench/bridge_six_cell_timing.jl` | n/a | `cf058168b` | n/a | 0.002791 s | **7.2×** vs TMB | #803 bank | `has_receipt` |
| DRModels | `drm-bridge-biv-rho12` | biv_gaussian rho12 n=180 | same | n/a | `cf058168b` | n/a | 0.001319 s | **15.2×** | #803 | `has_receipt` |
| DRModels | `drm-bridge-gamma` | Gamma FE n=180 | plus5 | n/a | `cf058168b` | n/a | 0.000808 s | **19.8×** | #803 | `has_receipt` |
| DRModels | `drm-bridge-beta` | Beta FE n=180 | #372 | n/a | `cf058168b` | n/a | 0.001356 s | n/a | R FAIL fenced | `has_receipt` (Julia abs) |
| DRModels | `drm-missing-gauss-n1000` | Gaussian locscale + missing y | `…/fixtures/missing_gauss_n1000.csv` | n/a | `e9d50a110` | tmb 0.033 | julia 0.001177 | **28.0×** | shared CSV; Δll≈2e-4 | `has_receipt` |
| DRModels | `drm-sigma-re-G40` | Gaussian sigma~(1\|g) | `…/fixtures/sigma_re_G40.csv` | n/a | `e9d50a110` | tmb 0.113 | julia 0.001696 | **66.6×** | Δll≈0.065 disclosed | `has_receipt` |
| DRModels | `drm-tweedie-fe-n400` | Tweedie FE n=400 | `…/fixtures/tweedie_fe_n400.csv` | n/a | `e9d50a110` | tmb 0.061 | julia 0.023955 | **2.55×** | shared CSV | `has_receipt` |
| DRModels | `drm-phylo-beta-p128` | Beta phylo p=128 m=4 | tip abs | n/a | `e9d50a110` | n/a | 0.023632 s | n/a | tip abs Mac | `has_receipt` |

## Count

DRModels: **18** `has_receipt` (was 10). Two slots remain toward 20.
