# 2026-11-16 — bridge gradient source matches the vector (#1402)

## What changed

`_bridge_flatten` copied `bridge_diagnostics(fit).grad_source` onto the top-level
payload key even when it did not attach `"gradient"`. That scalar source is
`:forward` on a route with an objective and no `fit.nllgrad` (ForwardDiff of
the objective, magnitude only). drmTMB's `drm_julia_gradient_source()` reads the
top-level key as the provenance of the vector and aborts on `"forward"` with no
vector.

The top-level key is now written only when a vector was sent, or when the
scalar source is already `:none` or `:unavailable`. Otherwise the key is
omitted, which that R function reports as `"unknown"`. The nested diagnostics
dict still carries the scalar `max_abs_grad` and its real producer. `check_drm`
is unchanged.

## Why not `"none"` / `"unavailable"` for the forward route

`"none"` means the fit stores no objective. `"unavailable"` means an objective
is stored but no finite derivative came out. The closed-form Gaussian route has
an objective and a finite ForwardDiff magnitude, so neither label is true.
The live drmTMB test
(`tests/testthat/test-julia-diagnostics.R`, the no-stored-gradient case) expects
`source=unknown`, which is the absent-field result. Sending `"none"` or
`"unavailable"` would get past the abort and fail that assertion. This repo
cannot change that test.

## Verification

Julia 1.10.13, `BLAS.set_num_threads(1)`, `--project=test`:

- `test/test_bridge_option_passthrough.jl` — 34 passed, 0 failed (23 + 5 + 2 + 4)
- `test/test_twin_gap_569.jl` — 48 passed, 0 failed
- `test/test_check_drm.jl` — 82 passed, 0 failed

The drmTMB live test was not run here (no R / JuliaCall bridge in this
environment).

## Rose

The claim is the payload no longer pairs a producer `grad_source` with a
missing gradient vector. The Julia tests above are the evidence. No drmTMB
source was vendored. No `drm()` or `check_drm` default changed.
