# After-task: Poisson animal q2 boundary restarts (#1333)

Personas: **Noether** (Laplace optimizer / `_q2_boundary_restarts`), **Karpinski** (Julia 1.10 CI leg). No spawned subagents.

## What changed

`_fit_poisson_animal_intercept_slope_laplace` now calls `_q2_boundary_restarts` after the default (and optional `second_start`) L-BFGS run, mirroring `_drm_binomial_sd_regression` for correlated Binomial animal q2. Alternative starts use `_q2_alt_theta` on the two group-level SD designs; acceptance follows the shared gain / converged-flip rules. `fit.scales[:q2_restarts]` records whether restarts ran.

## Rose

Claim: nightly Julia **1.10** CI failed only on the Poisson animal cell of `test_q2_sd_simulate.jl`; Julia **1** shards passed on the same SHA. Fix is scoped to the Poisson independent animal intercept-slope route; q=4 engine baseline not re-benchmarked. No drmTMB GPL source; no parity fixture numbers changed.

## Verification

Linux cloud agent, `OPENBLAS_NUM_THREADS=1`, Julia 1.10.13, `julia +1.10 --project=test --check-bounds=yes`:

| `test/test_q2_sd_simulate.jl` | `main` @ pre-fix | branch `5118dfb0`+DoD |
|---|---|---|
| Pass / total | 77 / 78 (1× `fit.converged` fail, Poisson animal) | 78 / 78 |

`bench/run_sparse_tmb_nd.jl` not run (Poisson q2 SD outer loop only).
