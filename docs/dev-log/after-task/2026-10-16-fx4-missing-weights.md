# After-task: FX4 (a)(b) — missing predictor KeyError + weights MethodError

## 1. Goal

Sweep FX4 cells (a) and (b) on the DRModels pin: a raw Julia `missing` in a
predictor used to throw a StatsModels `KeyError`, and `drm(...; weights = w)`
used to throw a `MethodError` (sweep t5). Shannon, with Noether (engine
hygiene) and Boole (named errors). No subagents.

Refs #1089, #1111, #931.

## 2. Implemented

- **#1089** `_design` runs `_prepare_predictor_table` first. An actual `missing`
  in a predictor is an `ArgumentError` naming `drm_listwise`. drm still does
  not drop incomplete predictor rows.
- **#1111** A `Union{Missing,T}` predictor with no missings is narrowed with
  `identity.` so StatsModels keeps it numeric (the dummy-coded-missing-as-a
  factor bug).
- **#931 option (b)** Every `drm` method takes `weights = nothing`.
  Fixed-effect Gaussian / Poisson / Gamma / Beta multiply the per-row
  log-likelihood; integer weights match row duplication. Random-effect,
  structured, REML, joint-`mi()`, and other-family routes refuse with a named
  `ArgumentError` (never a `MethodError`). `DrmFit.prior_weights` stores the
  vector; `weights(fit)` returns `nothing` when unweighted (not `ones(nobs)`).

## 3a. Decisions and rejected alternatives

- Did not auto-drop missing predictor rows inside `drm` (owner #1089: named
  refusal + `drm_listwise`).
- Did not implement RE / REML / other-family weights (owner #931 option (b)).
- Did not keep the `ones(nobs)` stub: an unweighted fit must not pretend it
  was weighted with ones.

## 4. Verification

- Red-proof: `test/test_fx4_missing_predictor_weights.jl` failed on the pin
  (KeyError, MethodError, `ones()` stub) and passes after the fix.
- `Pkg.test()` after the fix.

## 5. Rose audit

- Claims are the named error, the Union{Missing} numeric path, and the FE
  weight identity — each has a test. No speed/accuracy headline.
- License boundary: no drmTMB GPL source vendored.
- Rosetta `weights(fit)` row updated from "planned (parity gap)" to the
  implemented accessor. Scope honesty: RE weights remain refused.
