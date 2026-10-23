# 2026-10-08 — non-finite inputs and the converged backstop

Closes #1009, #1012, #1019, #1021. Personas: Shannon coordinating; Noether (constructor backstop, no change to the Laplace objective); Pat (the error names the argument and the row); Hopper (formula bridge and the two q2 bridge entries). No subagents were running.

## What changed

`_require_finite_inputs` (`src/finite_inputs.jl`) is the only checker.

- Every formula route builds its response and design in `_coerce_response_column` / `_design`, so Student `(1 | g)`, temporal, phylo, mixed `(1 | g)`, bivariate, cross-family `drm`, locscale and the formula bridge inherit the check without a per-family patch.
- `Inf` / `-Inf` in a response is an error. `missing` and `NaN` stay the observed-rows pattern.
- Predictors are checked as named columns and again on the built design matrix (so `log(0)` is caught). Offsets are checked in `_offset_vector`. Coordinates are checked before a spatial distance matrix is built (Gaussian, Poisson, bivariate q2/q4, spatial bootstrap).
- `fit_mixed_family`, `drm_bridge_q2_phylo` and `drm_bridge_q2_known_precision` call the same helper because they do not go through `_design`.
- `associate_pairs` refuses a margin whose `nobs` is shorter than the stored response, refuses non-finite margin `y` / `mu` / `sigma`, and throws if the pair log-likelihood at the optimum is non-finite or `-floatmax`.
- The `DrmFit` inner constructor clears `converged` unless the log-likelihood is finite and above the `-1e15` sentinel and every coefficient is finite. `is_converged` uses that same sentinel and coefficient rule, and is still stricter for a collapsed Gaussian residual scale.

## Behaviour change

Inputs that used to return a fit with `loglik` of `-Inf`, `NaN` or `-1e18` and `converged = true` now raise `ArgumentError` before the optimiser runs. A fit object built with a sentinel log-likelihood stores `converged = false`.

## Verification

Julia 1.10.12, `julia --project=.`, one process per batch. Every `@testset` below passed with zero failures. The temporal and mixed files need `StableRNGs` from the test environment; they were run after adding that package locally and the `Project.toml` change was not kept.

| Suite | Pass | Total | Time |
|---|---:|---:|---:|
| `test_triage_h_nonfinite.jl` | 60 | 60 | 1m 02.7s |
| `test_sentinel_fit_level.jl` | 18 | 18 | 19.6s |
| `test_numerical_guards.jl` | 39 | 39 | 13.1s |
| `test_spatial_coord_poisson.jl` | 11 | 11 | 24.2s |
| `test_relmat_counts.jl` | 19 | 19 | 4.2s |
| `test_gaussian_bivariate_q4_structured.jl` | 41 | 41 | 34.6s |
| `test_student.jl` | 5 | 5 | 3.5s |
| `test_student_re.jl` | 5 | 5 | 20.5s |
| `test_student_ordinary_laplace.jl` | 19 | 19 | 9.0s |
| `test_bridge.jl` | 146 | 146 | 1m 29.0s |
| `test_bridge_q2_direct_export.jl` | 179 | 179 | 19.9s |
| `test_associate_pairs.jl` | 43 | 43 | 32.5s |
| `test_temporal_ar1.jl` | 130 | 130 | 1m 14.8s |
| `test_temporal_ou.jl` | 95 | 95 | 11.3s |
| `test_temporal_homtoep.jl` | 180 | 180 | 41.8s |
| `test_temporal_boundary.jl` | 24 | 24 | 3.7s |
| `test_mixed_family.jl` | 78 | 78 | 1m 58.4s |
| `test_sentinel_mixed_family.jl` | 22 | 22 | 9.3s |

1114 passed, 0 failed. Times are the `@testset` times; a file with several testsets is the sum of those times. `bench/run_sparse_tmb_nd.jl` was not re-run. The Laplace objective was not edited.

## Review round (2026-10-09)

The predictor check in `_design` now runs only on rows whose response is observed, so a missing response and a `NaN` predictor on that same row fit with `nobs = n - 1` (review case F). Named checks cover `meta_V` variances, random-slope covariates read from `data`, the phylogenetic structured slope, user `K`/`A` matrices, `fit_phylo_interaction`, and the binomial/beta-binomial failures column. Coordinate tables are materialised before the finite check. Tree branch lengths were already rejected by `_phy_branch_length` (the edge is named). The assembled length vector is checked again by `_phy_require_finite_lengths` in `sparse_phy.jl`, so the experimental location-only loader can `include` that file without `finite_inputs.jl`. `predict(newdata)` without a response checks every row, so a `NaN` predictor there is an error. The bare `catch` in the column walker now rethrows anything that is not a missing-column error.

## Rose

Claim: the four issues are fixed by one helper plus the constructor backstop, and the sibling routes listed above are covered because they call that helper or `_design`. Evidence: the table above, including one repro per issue in `test/test_triage_h_nonfinite.jl` and the existing route suites. No drmTMB source was vendored. No speed or parity number was changed. Observation `weights` are not a `drm` argument (`weights(fit)` is all ones); the helper rejects a non-finite weights vector and the test calls it directly. That is a gap in the public API, not a silent NaN fit.

## Full suite (2026-10-09, Julia 1.10.12, `--check-bounds=yes`)

`Pkg.test()` ran the discovered files through `test_q4_reml_vcov.jl` and then stopped. That file's native `vcov` pin failed with `maximum(relerr) = 0.0021784715166058583` against `1e-3`. The same number, on the same line, fails on commit `32e2d0c94` with these edits stashed, so it is not caused by the finite-input checks. The experimental optimizer file, which the first `Pkg.test()` aborted on (`_require_finite_array` undefined inside the standalone `sparse_phy.jl` include), passes 12/12 after the length check moved into `sparse_phy.jl`.

The 113 files after `test_q4_reml_vcov.jl` were included in a continuation process. Files that load `StableRNGs` or other test-only packages were re-run in an environment that has those packages. No `@test` failed in that continuation. `test/test_triage_h_nonfinite.jl` is 87/87.
