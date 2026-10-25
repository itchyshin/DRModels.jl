# After-task: bivariate training-schema cache (#1000)

## Goal

Bivariate `predict` / `predict_parameters` rebuilt factor contrasts from `newdata`. A level subset was refused (or, with the same column count, could be silently wrong). Univariate fits already cached the training schema (#609).

## Implemented

`BivariateDrmFormula` gained a `schema_cache` field, populated by `_biv_design` at fit time on the residual and structured bivariate Gaussian routes and the residual bivariate Student-t route. `predict` and `predict_parameters` pass that cache instead of `nothing`.

## Evidence

`test/test_bivariate_schema_cache_1000.jl` (13/13) plus `test/test_predict.jl` (4/4) on Julia 1.10.12 with `--check-bounds=yes`. A three-level factor, predict on the `"b"` rows only, matches the full-data prediction on those rows; an unseen level still errors. The same subset check is run on a bivariate Student-t fit.

## Not done

Non-Gaussian univariate fitters that still skip `schema_cache` (the later comment on #1000) are unchanged. The column-name guard remains as a backstop when a route does not populate the cache.

No drmTMB GPL source was used.
