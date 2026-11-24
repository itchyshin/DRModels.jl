# 2026-11-19 — ZeroOneBeta with a missing response: implied `zoi` / `coi` in `predict_parameters` (refs #1389)

Perspectives: **Shannon** (slice), **Noether** (prediction path), **Hopper** (drmTMB 0.8.1 reference values from the issue), **Rose** (claim vs evidence). No subagents.

## What changed

- `predict_parameters` (`src/gaussian_core.jl`): for a `ZeroOneBeta` fit, a `zoi` or `coi` coefficient block with no formula in `fit.formula.forms` now gets the implied intercept-only predictor (`ConstantTerm(1)`). `bf()` stores only the formulas the user wrote, so `bf(y ~ x)` returned no `:zoi` / `:coi`.
- Effect: `_full_means_and_scales` (which rebuilds per-row parameters on the original data when the response has `missing` values) no longer throws `KeyError: key :zoi not found`; `predict(fit, newdata; type = :mean)` works on complete data.
- Test: `test/test_1389_zeroonebeta_missing_response.jl`. On `main` (`850a20bd0`) the five testsets error with `KeyError: key :zoi not found`; with the fix 26 of 26 pass (Julia 1.10.13).

## Evidence

- Fixed-effect fits on the issue's deterministic data: `nobs = 50`; logLik 2.533589 (`y ~ x`) and 2.585723 (`y ~ x`, `sigma ~ x`) equal the drmTMB 0.8.1 values recorded in #1389 (asserted to 1e-5). The reference numbers were not re-run in R here.
- The fit with no `zoi`/`coi` formula equals the fit with `zoi ~ 1`, `coi ~ 1` written out (`loglikelihood`, `fitted`, `predict_parameters` on link and response scale, `predict(type = :mean)`, all `==`).
- The random-intercept fit now runs and returns `nobs = 50` with a finite logLik. The issue records that the package integrates that intercept by adaptive quadrature and drmTMB by Laplace, a known gap; no parity claim is made for it.
- Related tests run locally and unchanged: `test_predict_parameters`, `test_zeroonebeta`, `test_zeroonebeta_contract`, `test_zeroonebeta_mean_ri`, `test_zeroonebeta_sd_regression`, `test_runtests_include_list`, `test_shard_layouts`. Full suite not run.

## Overlap

Draft PR #1380 changes the same `forms = Dict(f.forms)` line of `predict_parameters` (it adds `nu` as well). Whichever lands second needs a trivial conflict resolution; the two changes give the same `zoi`/`coi` result. #1380 does not cite #1389.

## Rose

**Claim:** a ZeroOneBeta fit with missing responses no longer throws, and prediction returns the estimated `zoi` and `coi` when no formula was written. **Evidence:** the test above. **Scope honesty:** no change to the likelihood, the optimiser, any tolerance or any existing test; `nu` for Student/SkewNormal/Tweedie is not touched here (see #1380).
