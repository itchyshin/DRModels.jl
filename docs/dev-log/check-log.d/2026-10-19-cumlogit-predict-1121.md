## 2026-10-19: CumulativeLogit predict (#1121)

- Fix: `_predict_design` (`src/gaussian_core.jl`) drops `(Intercept)` for the `CumulativeLogit` `mu` block, matching `src/cumulative.jl`.
- Test: `test/test_cumlogit_predict.jl` (errored on `main`, 4 of 4 with `ArgumentError`; passes after the fix).
- Local: full `Pkg.test()` on Julia 1.10.5, see the PR for the result.
