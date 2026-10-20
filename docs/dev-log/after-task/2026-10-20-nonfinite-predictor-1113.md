# After-task: NaN/Inf predictor with `(1 | g)` (#1113)

## Goal
A NaN or Inf in a numeric predictor on a row whose response is observed must not return a `DrmFit` of starting values (Poisson, NB2, Binomial, Student) or an unnamed LAPACK / ForwardDiff error (Gaussian, Gamma).

## Implemented
`_check_finite_predictors` (`src/gaussian_core.jl`), called from `_design` on the fit path only. It walks the RHS columns, skips the response and `missing` cells, and throws an `ArgumentError` naming the column, the value and the first offending row. Rows whose response is missing or NaN are not checked, so they are still dropped with the response. Predict paths (`store_schema = false`) are unchanged, because the post-fit full-row prediction also scores the dropped rows. The false in-code statements in `gaussian_core.jl` and `missing_data.jl` are corrected.

## Evidence
`test/test_nonfinite_predictor_re.jl`: six families, NaN / Inf / -Inf, Float32, and the dropped-response case. 52 of 57 checks fail on main; 57 of 57 pass with the fix.

## Not done
`missing`-coded predictors (#1089) and the wider non-finite-input work in draft #1045 are untouched; the two overlap in `_design` and will need a merge.

No drmTMB GPL source was used.
