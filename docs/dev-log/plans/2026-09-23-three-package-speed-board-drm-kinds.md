# DRM speed-board kinds addendum (2026-09-23)

Companion to the three-package speed board and
`2026-09-23-speed-report-20x3-diversity.md`.

Authoritative numbers: `docs/dev-log/evidence/2026-09-23-speed-kinds-toward20/`.

## Wave A: Mac Studio (+8; 10 → 18)

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

## Wave B: Totoro diversity first-wave (+10)

| package | cell_id | kind | wall_med_s | host | sha | notes | status |
|---|---|---|---:|---|---|---|---|
| DRModels | `drm-phylo-poisson` | D-phylo-pois | 0.022278 | totoro | `e9d50a110` | Julia abs | `has_receipt` |
| DRModels | `drm-phylo-nb2` | D-phylo-nb2 | 0.034751 | totoro | `e9d50a110` | Julia abs | `has_receipt` |
| DRModels | `drm-phylo-binomial` | D-phylo-binom | 0.023230 | totoro | `e9d50a110` | Julia abs | `has_receipt` |
| DRModels | `drm-phylo-gamma` | D-phylo-gamma-beta | 0.071680 | totoro | `e9d50a110` | Julia-only fence | `has_receipt` |
| DRModels | `drm-h2h-q4-vs-tmb-p1000` | D-gauss-q4-phylo | 20.814729 | totoro | `e9d50a110` | TMB pair owed | `has_receipt` |
| DRModels | `drm-crossed-binomial` | D-crossed-family | 0.037384 | totoro | `e9d50a110` | Julia abs | `has_receipt` |
| DRModels | `drm-biv-gauss-rho12` | D-biv-gauss | 0.057182 | totoro | `e9d50a110` | Julia abs | `has_receipt` |
| DRModels | `drm-profile-ci-locscale` | D-profile-ci | 0.015969 | totoro | `e9d50a110` | Julia abs | `has_receipt` |
| DRModels | `drm-animal-gauss` | D-animal | 0.017705 | totoro | `e9d50a110` | Julia abs | `has_receipt` |
| DRModels | `drm-lss-sd-slope` | D-lss | 0.002877 | totoro | `e9d50a110` | Julia abs | `has_receipt` |

## Count

DRModels: **28** `has_receipt` (floor 10 + Wave A 8 + Wave B 10). Diversity
aim ~20 cleared. Soft A3 skipped. No public speed claim.
