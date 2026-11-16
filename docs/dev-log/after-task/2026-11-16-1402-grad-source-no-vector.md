# 2026-11-16 — bridge gradient source is the vector's producer (#1402)

## What changed

`_bridge_flatten` was copying `bridge_diagnostics(fit).grad_source` onto the
top-level payload key. That scalar is the producer of `max_abs_grad`
(`_check_max_abs_grad`), not of the `"gradient"` vector. The vector has one
producer, `fit.nllgrad`, so a shipped vector is always `"stored"`. The scalar
can differ: a `LocScaleObjective` is checked first (`:locscale`, analytic
gradient in permuted order), and a non-finite `nllgrad` magnitude falls
through to `:finite` or `:unavailable` while the vector itself was still being
shipped.

The vector is now attached only when every entry is finite. The top-level
`grad_source` is `"stored"` exactly then. Otherwise the key echoes `"none"` or
`"unavailable"` when the scalar source says so, and is omitted in every other
case (drmTMB reports a missing key as `"unknown"`). The nested diagnostics
dict still carries the scalar `max_abs_grad` and its real producer. `check_drm`
is unchanged.

No public location–scale route stores `nllgrad`. `_fit_locscale_frontend`
attaches the objective with `_withnll(fit, LocScaleObjective(...))`, whose
gradient argument defaults to `nothing`. There is no route on which a shipped
vector and a `:locscale` scalar occur together, so no such route test was
added.

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

- `test/test_bridge_option_passthrough.jl` — 41 passed, 0 failed (30 + 5 + 2 + 4)
- `test/test_twin_gap_569.jl` — 48 passed, 0 failed
- `test/test_check_drm.jl` — 82 passed, 0 failed
- LocScale gradient files, 0 failed: `test_locscale_whitened.jl` 284;
  `test_locscale_bootstrap_refit.jl` 15; `test_locscale_grad.jl` 9;
  `test_locscale_fit.jl` 24 (5 + 3 + 6 + 10); `test_locscale_gamma_e2e.jl` 6;
  `test_locscale_precision_derivatives.jl` 28 (24 + 4);
  `test_locscale_profile_nan.jl` 14 (6 + 4 + 4);
  `test_optimizer_robustness.jl` 22; `test_nonconst_sigma_re.jl` 21;
  `test_phylo_locscale.jl` 14; `test_locscale_structured.jl` 29
- `test/test_locscale_profile.jl` was included. Its two testsets stay behind
  `DRM_SLOW_TESTS=1` (the file's own gate) and did not execute.

The drmTMB live test was not run here (no R / JuliaCall bridge in this
environment).

## Rose

The claim is: a shipped `"gradient"` is finite and labelled `"stored"`, and a
missing vector is not labelled with a scalar producer other than `"none"` or
`"unavailable"`. The Julia tests below are the evidence. No drmTMB source was
vendored. No `drm()` or `check_drm` default changed.
