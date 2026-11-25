# 2026-11-24 — Parity SE and vcov at the compare.jl defaults (refs #1503)

Perspectives: **Shannon** (slice), **Hopper** (fixture comparison), **Rose** (claim vs evidence). No subagents.

## What changed

- New always-on test `test/test_parity_vcov_defaults.jl`. For every in-cohort fixture that stores `[vcov]`, it refits with the same dispatch as `test/parity/runparity.jl` (`drm` / `drm_bridge`) and compares through `compare_fit` / `compare_bridge` after removing `rtol_vcov`, `atol_vcov`, `rtol_se`, and `atol_se` from the loaded `[tol]`. The keyword defaults in `test/parity/compare.jl` then apply: `rtol_vcov = 1e-3`, `atol_vcov = 1e-8`, `rtol_se = 1e-3`, `atol_se = 1e-8`.
- Other `[tol]` keys (log-likelihood, AIC, coefficients) are left untouched. No fixture file, no existing test, and no default in `compare.jl` was edited.
- **EXCLUDED from vcov only** (excluded per Shinichi 24 Nov 2026): `meta-analysis-V` (`|Δ| = 1.4619471914728503e-8` on `vcov[mu_(Intercept),sigma_(Intercept)]`) and `robust-student` (`|Δ| = 2.0435463875702732e-8` on `vcov[mu_x,nu_(Intercept)]`). All other parity axes, including SE, still run for both. Default `atol_vcov = 1e-8` unchanged. `xfam-external-gllvm` stays out of the cohort, as in `runparity.jl`.

## Evidence

`julia --project=test -e 'using Test; include("test/test_parity_vcov_defaults.jl")'` on Julia 1.13.4, `OPENBLAS_NUM_THREADS=1`: 10 passed in the default-override testset and 109 passed in the fixture testset (119 total, 0 failed). The fixture half took 1m56s.

Re-measured on the same Julia at `a029c2710`, vcov keys stripped, before this test existed:

- `meta-analysis-V`: `vcov[mu_(Intercept),sigma_(Intercept)]` `|Δ| = 1.4619471914728503e-8` (symmetric entry the same). drmTMB `2.449811252561793e-6`, DRModels.jl `2.4351917806470644e-6`.
- `robust-student`: `vcov[mu_x,nu_(Intercept)]` `|Δ| = 2.0435463875702732e-8` (symmetric entry the same). drmTMB `-9.305856632734159e-6`, DRModels.jl `-9.326292096609861e-6`.

Both sit between the default `atol_vcov = 1e-8` and the fixture's existing `atol_vcov = 3e-8`. No new tolerance was added. The gated suite still uses the fixture overrides; this test does not adopt them for the vcov/SE axes.

## Rose

**Claim:** stored fixture covariances and standard errors are asserted at the compare.jl defaults for 16 in-cohort cells; the two named cells are **EXCLUDED from vcov only** per Shinichi 24 Nov 2026, with measured deltas recorded and no tolerance widening. **Evidence:** `test/test_parity_vcov_defaults.jl`, the two symmetric cross-terms above. **Scope honesty:** likelihood, optimiser, convergence rules, and default tolerances unchanged. Bridge fixtures have no `[se]` block, so no standard-error numbers were invented for them.
