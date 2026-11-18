# #938: coefficient-1 `offset()` in the mean formula for every non-Poisson family

## 1. Goal

Implement the owner decision of 25 Oct 2026 on #938: a working `offset()` in the
mean formula (coefficient fixed at 1) for NegBinomial2 and the other non-Poisson
families, reviving the closed PR #1096 plan; keep refusing an offset in the
`sigma` formula, as drmTMB does; then the bridge change and an Owls parity row.

## 2. Implemented

- Shared helpers in `src/poisson.jl`: `_eta_with_offset`, `_add_offset`,
  `_offset_start` (start value only), `_offset_for_fit`, `_refuse_offset_route`.
- Every univariate non-Poisson family applies the offset on its fixed-effects
  mean kernel: NegBinomial2, TruncatedNegBinomial2, TruncatedPoisson, Gamma,
  LogNormal, Student, SkewNormal, Beta, ZeroOneBeta, Binomial, BetaBinomial,
  Tweedie, CumulativeLogit. NegBinomial2 also with `zi`/`hu` and one ordinary
  `(1 | g)` (`:LA` GHQ kernel). `_fit_scale_ordinary_laplace` (NB2, Gamma, Beta
  under `marginal = :Laplace`) threads it through `_phylo_mean_laplace_nuisance_fg`.
- Gaussian: identity link, so `offset(z)` is fitted as `y - z` and `z` is added
  back to the fitted means and observed response (fixed-effects mean only).
- Refusals: `_require_offset_supported` now refuses an offset in every non-mean
  formula for every family (drmTMB parity) and leaves the mean to the routes;
  each family refuses unsupported routes by name; `_design` refuses any offset
  still present (for example `x * offset(z)`), so no route can estimate it.
- Bridge: `offset` left `_BRIDGE_REJECT_CALLS`; `_bridge_xlate` passes
  `offset(column)` or `offset(transform)` to the marker, and the label renderer
  drops it from the public coefficient columns.
- Owls parity: `test/parity/gen_owls_offset_938.R` (drmTMB run, never vendored;
  the AGPL data are written outside the repo) and
  `test/test_offset_owls_parity_938.jl` (skips without `DRM_OWLS_CSV`).

## 3a. Decisions and Rejected Alternatives

- Not added: the `offset =` keyword and the NB2 `sigma` offset from #1096; the
  decision asks for neither, and keeps sigma offsets refused.
- Gaussian is limited to the fixed-effects mean: the response shift is exact on
  other routes too, but their fitted objects were not audited for stored
  copies of `y`.
- The `_design` backstop was kept from #1096 so a nested or unanticipated
  offset can never become a free coefficient.

## 4. Files Touched

`src/{poisson,input_validation,gaussian_core,negbinomial,truncated_poisson,gamma,
lognormal,student,skewnormal,beta,zeroonebeta,binomial,betabinomial,tweedie,
cumulative,ordinary_laplace,sparse_laplace_glmm,bridge}.jl`;
`test/test_offset_fixed_coef_938.jl`, `test/test_offset_owls_parity_938.jl`,
`test/parity/gen_owls_offset_938.R`, `test/test_lane_j3_input_validation.jl`,
`test/test_bridge_silent_sweep.jl`; `docs/src/reference/model-specification.md`;
`news/938-offset-fixed-coefficient.md`; this file and its check-log row.

## 5. Checks Run

Julia 1.12.7 on Totoro, one process per file, BLAS 1 thread:
`test_offset_fixed_coef_938.jl` 138/138 (red on origin/main bd683e28d: all 14
family testsets error on the #1057 refusal); `test_offset_owls_parity_938.jl`
17/17 with `DRM_OWLS_CSV` set; `test_lane_j3_input_validation.jl` 126/126;
`test_lane_j4_predict_numerics.jl` 122/122; `test_twin_gap_727.jl` 15/15;
`test_triage_h_nonfinite.jl` 87/87; `test_count_lognormal_contract.jl` 115/115;
`test_bridge_silent_sweep.jl` 37/37; `test_bridge_formula_translation.jl` 54/54;
`test_ordinary_laplace.jl` 335/335; the Binomial, CumulativeLogit, Gamma/Beta,
TNB2, TruncatedPoisson/SkewNormal, Student, ZeroOneBeta and Tweedie contract
files, `test_poisson.jl`, `test_lognormal.jl`, `test_truncated_nb.jl`,
`test_nb2_hurdle_nan.jl`, `test_student_ordinary_laplace.jl` and
`test_1121_cumulative_predict.jl` all pass. `test_lane_n4_offset_slopes.jl`
is 40 pass, 1 error (a LineSearches `b > a` assertion in the Gamma correlated
`(1 + time | g)` GHQ fit); origin/main bd683e28d gives the same 40/1, so it is
pre-existing and untouched here. drmTMB numbers: 8aafcc2c via
`pkgload::load_all`.

## 6. Tests of the Tests

The new test file was run on origin/main first: all 14 family testsets errored
on the #1057 refusal (red). Each family test compares a constant offset with the
offset-free fit (intercept moves by exactly -log 2, logLik equal), so a free or
dropped offset fails it.

## 7a. Issue Ledger

#938 stays open for: offsets with random effects on routes other than those
listed (crossed, slopes, structured, VA, coupled location-scale, Gaussian random
effects, Binomial `:Laplace`); and the drmTMB side of the bridge
(`R/julia-bridge.R` still refuses `offset()` for `engine = "julia"`).

## 8. Consistency Audit

`offset` docstring, NegBinomial2 docstring and the model-specification page now
state the same route list. `predict(newdata)` already re-applied a fitted
offset (#1006) and now does so for every family.

## 9. What Did Not Go Smoothly

The first green run used `coefnames`, which is not exported, and a SkewNormal
generator that set the location instead of the mean; both were test-side bugs.

## 10. Known Residuals

No full `Pkg.test()` was run; only the offset-related and bridge files.

## 11. Team Learning

A constant-offset identity check (intercept shift, equal logLik) is a cheap,
family-agnostic oracle for "the offset is held at 1".
