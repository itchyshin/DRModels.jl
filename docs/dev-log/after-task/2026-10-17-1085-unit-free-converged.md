# After-task: unit-free `converged` at a tiny residual SD (#983 / PR #1085)

## 1. Goal

CI run 38846590757 on PR #1085 head `265b6d39` was red on shard 4/4 for Julia
1.10 and Julia 1, at `test/test_converged_unit_free_983.jl:21`: not every seed
in `20275928:20275939` reported `converged` at residual SD `1e-5`. Fix the
root cause in the fitter, not the test. Shannon, with Noether (engine
contract) and Rose (claim vs evidence). No subagents.

## 2. Implemented

- **`_fit_fixed_gaussian`** no longer ANDs `!Optim.iteration_limit_reached(res)`
  into the Newton-decrement stationarity path. LBFGS can exhaust the default
  1000-iteration cap on noise-limited steps *at* the optimum of a tiny-scale
  Gaussian: seed `20275937` at the issue's SD `exp(-9.2)` took 1000 iterations
  with decrement `~4e-23`. A cap-hit far from the optimum still has a large
  decrement and stays `false`.
- **`_newton_decrement_small`** computes `½ g'H⁻¹g` after diagonal
  equilibration. The mean-block curvature scales like `1/σ²` while the log-σ
  curvature stays `O(n)`, so an unscaled Cholesky can fail on some LAPACK
  builds even though `H` is theoretically PD (rcond `~1e-10` at SD `1e-5`).
  The decrement itself is invariant to that scaling.

Not touched: `drm_optim_converged`, `_scale_runoff`, `_SCALE_RUNOFF_LOG_SPREAD`,
`_vcov_from_hessian` / `_VCOV_RTOL` (the "flat coordinate 3" warning at these
seeds is a false relative-diagonal flag, #936; out of this slice), the REML
sibling `_fit_fixed_gaussian_reml`, and every drmTMB `#1513` source file.

## 3a. Decisions and rejected alternatives

- **Not a genuine variance boundary.** Every seed in the forcing set has
  `σ̂ ≈` residual SD, log-σ Hessian diagonal `≈ 2n = 200`, and a tiny decrement.
  The vcov_guard warning (`flat_coordinates = [3]`, rcond `1e-8`–`1e-10`)
  compares `diag(H)[σ]` to `max diag(H)` and so flags the *well-identified*
  scale when the mean block is huge. That is #936, not a boundary this test
  should accept as `converged = false`.
- **Do not raise `iterations`.** That treats the symptom; a slower BLAS or a
  slightly smaller SD would hit the new cap.
- **Do not use `pinv(H)` for the decrement.** The same relative-eigenvalue
  cutoff that flags coordinate 3 would zero the scale direction, so a
  not-yet-stationary scale gradient would look small.
- **Keep the helper's non-PD rejection.** Equilibration does not change the
  existing contract cases (large decrement, indefinite `H`, NaN `H`).

## 3b. Evidence

Local Julia 1.10.12, OpenBLAS, one BLAS thread, on `265b6d39` before the fix:

| seed | SD | iters | decrement | `converged` on `265b6d39` |
|---|---|---|---|---|
| 20275937 | exp(-9.2) | 1000 | 4.3e-23 | `false` (iteration-cap veto) |
| 20275928–36, 38–39 | exp(-9.2) | 7–257 | ≤1.3e-22 | `true` |
| 20275928–39 | 1e-5 | 7–575 | ≤7e-16 | `true` locally |

CI run 38846590757 failed the `1e-5` assertion (line 21) on both Julia
versions: the same cap veto, with platform-dependent iteration counts. Line 19
(`exp(-9.2)`) passed on CI and failed here — the same defect, different seed.

After the fix, on Julia 1.10.12 (OpenBLAS, one BLAS thread):

- `test/test_converged_unit_free_983.jl`: 14 pass (6 + 8), including the
  ill-conditioned-Hessian helper cases.
- `DRM_TEST_SHARD=4/4 julia --project=test test/runtests.jl`: exit 0 in 857 s
  (93 of 373 files; the #983 file 6+8 pass).

## 4. Rose audit

- **Claim vs evidence.** "All twelve seeds report `converged`" is a measured
  local result after the fix, not an extrapolation. The 2.18× / logLik −256.51
  engine baseline is not re-claimed here (this slice does not touch the q=4
  path).
- **Scope honesty.** Only `_fit_fixed_gaussian` and `_newton_decrement_small`.
  REML, vcov_guard calibration, and drmTMB `#1513` are out of scope.
- **License.** No drmTMB GPL source vendored.
- **Doc drift.** NEWS fragment `news/1085-unit-free-converged-iter-cap.md`;
  no README / HANDOVER / AGENTS edit.

## 5. Follow-up

- `#936` / `#1049`: the vcov_guard relative-diagonal test still warns
  "variance boundary" on these well-identified tiny-σ fits.
- `_fit_fixed_gaussian_reml` still uses the absolute `g_tol` only.
