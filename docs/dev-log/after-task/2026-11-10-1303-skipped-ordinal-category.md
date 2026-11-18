# After-task: CumulativeLogit refuses a skipped category (#1303)

Personas: **Noether** (fit-path guard), **Hopper** (drmTMB message intent), **Pat** (the error names the missing and observed levels). No spawned subagents.

## What changed

`drm(..., CumulativeLogit())` took `K = maximum(y)` after checking only that every code was an integer `≥ 1`. A response in `{1, 2, 4}` therefore allocated a cutpoint for level 3, drove that increment to `−Inf`, and returned a fit with two identical thresholds.

The check now sits in `_drm_cumulative_fit`, after the integer/`K ≥ 2` check and before the fixed, random-intercept, random-slope, phylo, and crossed routes. A hole in `1:K` is an `ArgumentError` that names the missing level(s) and the observed levels, and points at dropping unused levels or combining sparse categories. Contiguous codes still fit, including `K = 2` when both codes occur.

## drmTMB

Read-only, on `itchyshin/drmTMB` `main` (`R/drmTMB.R`, `prepare_ordinal_response` / `validate_ordinal_codes`). Nothing in that repo was edited, and no GPL source was copied here.

drmTMB already refuses this case. For a numeric score it builds levels `1:max(y)` and aborts when any count is zero: every ordinal category must appear at least once, the empty category is named, and the hint is to drop unused levels or combine sparse categories. The Julia error follows that intent and also names the observed levels, which this issue asked for.

drmTMB additionally requires at least three categories. This family already documents and tests `K ≥ 2`. That rule was left as it is. A contiguous two-level response still fits, and the new test pins that.

## Other ordinal paths

`CumulativeLogit` is the only ordinal response family. `CategoricalLogit` is a nominal predictor family and does not call this fitter. The ordinal missing-predictor path (`_joint_finite_numeric_levels`) already requires contiguous `1:K` and was not changed.

## Rose

The refusal is what the tests measure: the issue MRE throws `ArgumentError` with missing level `3` and observed levels `1, 2, 4`; a filled `1, 2, 3` response returns a two-cutpoint fit. No parity fixture, tolerance, or coverage exclusion was touched. The two former `@test_broken` rows in `test_cumulativelogit_contract.jl` are real tests now.

## Verification

Julia 1.13.3, `BLAS.set_num_threads(1)`, `julia --project=.`:

| File | Result |
|---|---|
| `test/test-fix-1303-skipped-category.jl` | 17/17 |
| `test/test_cumulativelogit_contract.jl` | 128/128 |
| `test/test_cumulative.jl` | 4/4 |
| `test/test_crossed_cumulative.jl` | 7/7 |
| `test/test_cumlogit_nan.jl` | 77/77 |
| `test/test_1121_cumulative_predict.jl` | 23/23 |
| `test/test_cumlogit_aghq.jl` | 15/15 |
| `test/test_cumlogit_phylo.jl` | 14/14 |
| `test/test_cumlogit_ranef.jl` | 14/14 |
| `test/test_cumlogit_sd_regression.jl` | 14/14 and 25/25 |
| `test/test_runtests_include_list.jl` | 17/17 |

`bench/run_sparse_tmb_nd.jl` was not run. This edit does not touch the q=4 Gaussian likelihood.
