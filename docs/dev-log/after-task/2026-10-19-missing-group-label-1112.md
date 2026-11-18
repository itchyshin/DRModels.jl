# After-task: missing/NaN random-effect grouping label (#1112)

## 1. Goal

A `missing` (or NaN) in the grouping column of `(1 | g)` was silently fitted as one
extra pooled level (`nobs` unchanged, `converged = true`); drmTMB drops those rows.

## 2. Implemented

`_group_index` (`src/gaussian_ranef.jl`), the single choke point every route uses to
read a raw grouping column, now throws an `ArgumentError` when any label is `missing`
or non-finite. It takes an optional `col` keyword so the message names the grouping
column; the 40-odd `_group_index(getproperty(data, grp))` call sites now pass
`col = grp`. Sites that pass an already-extracted label vector still refuse, with a
generic "a grouping column" message. The message points to `drm_listwise`.

## 3a. Decisions and rejected alternatives

- Refuse rather than drop. Both are allowed by the issue; dropping would need row
  filtering of y, X, weights, offsets and every structured matrix in every route, a
  far larger and riskier change than one check at the choke point.
- `±Inf` labels are refused too (same predicate as `drm_listwise`).

## 4. Verification

- New `test/test_1112_missing_group_label.jl`: before the fix 6 of 9 assertions
  failed (nothing thrown); after, 9/9 pass. Covers `missing` String labels, NaN in a
  numeric code column, Gaussian with and without sigma formula, Gamma and Poisson
  `:Laplace`, and the `drm_listwise` workaround (nobs 110, 12 levels).
- Every other `test_*.jl` file ran standalone in the test environment. Not passing,
  and identical on the parent commit `b1832c7`: `test_bridge_bootstrap_tree.jl`
  (2 assertions), `test_reml_q4_missing_response.jl` (`fit_reml.converged`),
  `test_crossed_tweedie.jl` (timed out at 50 min on both). Not run:
  `test_qgate_jet.jl` and the files `runtests.jl` excludes.
- Not run in full through `Pkg.test()`: it stops at the first failing file
  (`test_bridge_bootstrap_tree`, a pre-existing failure), so the per-file run above
  replaces it. CI is the authority for the shard run.
- Not exercised: the drmTMB-parity numbers in the issue (R not involved), the
  remaining routes (beta, binomial, negbinomial, student, lognormal, tweedie,
  cumulative, phylo/structured) beyond the shared choke point.
