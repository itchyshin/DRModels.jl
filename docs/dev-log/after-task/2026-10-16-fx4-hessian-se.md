# After-task: FX4 (c) — GasolineYield non-PD Hessian must not print SEs

## 1. Goal

Sweep t8 GasolineYield: DRModels.jl printed a full coefficient table of finite
SEs on an unconverged Beta loc-scale fit, and turned a NaN (non-PD) vcov into
`Inf` Std.Error cells. Shannon, with Fisher (inference display). No subagents.
Does not change `is_converged` (drmTMB #1554 methods choice).

## 2. Implemented

- `_display_se` withholds NaN SEs when `_hessian_se_unusable`: the fit is not
  converged, or `vcov` has NaN / negative diagonal entries.
- `show` prints one line: Wald SEs withheld (Hessian not PD, or not converged).
- `coeftable` uses the same `_display_se`, so the Std.Error column is NaN.

## 3a. Decisions and rejected alternatives

- Did not fold non-PD / unconverged into `is_converged` (#1554 owner choice).
- Did not change `stderror()` / `vcov()`: those remain the Hessian-based
  numbers (Inf at a boundary; NaN matrix when the guard withholds). Only the
  **print** path is gated, matching "must not print SEs".

## 4. Verification

- Red-proof `test/test_fx4_hessian_se_display.jl`: fails on the pin (Inf / finite
  SEs), passes after.
- `Pkg.test()`.

## 5. Rose audit

- Claim is display-only; no speed/accuracy headline.
- License boundary: no drmTMB GPL source.
- Twin: drmTMB print/summary treat `pdHess = FALSE` as SEs unavailable in the
  sibling PR. Scope honesty: Julia still may stop short of the R optimum; this
  slice only stops presenting SEs as usable.
