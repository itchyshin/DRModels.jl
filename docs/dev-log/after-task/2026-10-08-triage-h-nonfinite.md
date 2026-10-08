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

## Rose

Claim: the four issues are fixed by one helper plus the constructor backstop, and the sibling routes listed above are covered because they call that helper or `_design`. Evidence: `test/test_triage_h_nonfinite.jl` (one repro per issue and per sibling) and the existing route suites. No drmTMB source was vendored. No speed or parity number was changed. Observation `weights` are not a `drm` argument (`weights(fit)` is all ones); the helper rejects a non-finite weights vector and the test calls it directly. That is a gap in the public API, not a silent NaN fit.
