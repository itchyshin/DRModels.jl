# After-task: rebase #1079 (J3b) onto current main

## 1. Goal

Finish PR #1079. Rebase onto current `main` (through latest #1099 `fa34ee763`,
already on `b1832c7a6`) so `pull_request` CI can run, keep the PR a draft, and
keep `Closes` honest. Shannon; Noether on the SE contract. No subagents.

## 2. Implemented

- Rebased the two unique J3b commits onto `fa34ee763`:
  `e32f5a597` (J3b routes + fixtures + harness) and `82f38749c` (call sites
  pass `(θ, tol)` into the shared `#1099` `_se_grad` ladder).
- Conflict in `src/sparse_laplace_glmm.jl` was comments only. Kept `#1099`'s
  single `_se_grad` (with `_SE_MODE_TOL_USED` / `_warn_se_mode_fallback`) and
  J3b's `_SE_LOGSD_LO` / `sd_lo` on `_se_hessian_path`. No second `_se_grad`.
- Added the missing DoD files this branch never carried: `news/1079-j3b-poisson-crossed-se.md`,
  this after-task, and `docs/dev-log/check-log.d/2026-10-18-j3b-poisson-general-crossed.md`.
- PR body: **Refs #1073**, **Refs #1099**. No `Closes` — J3b is one lane of #1073
  and does not own #1088.

## 3. Evidence

Local: Julia 1.10.12, `--check-bounds=yes`, BLAS 1 thread.

| case | result |
|---|---|
| `test/test_j3_se_reference.jl` | 70/70 in 1m18.8s |
| `test/test_j3a_gradient_hessian_vcov.jl` | 45/45 |
| `test/test_finite_hessian_nan.jl` | 38/38 |
| `test/test_1088_nb2_crossed_runoff.jl` | 62/62 |
| full `julia --project=test --check-bounds=yes test/runtests.jl` (374 files) | **16,763 passed / 0 failed** (740 testsets), EXIT 0, 00:25:44–04:59:08Z |

License: fixtures are generated drmTMB numbers only. No GPL source.

## 4. Open / follow-ups

- #1073 stays open (J3c / J3d / remaining routes).
- #1080 is next: rebase onto this head, then retarget to `main`.
- Ancestor commit `9df3805e6` still says `Closes #1088` (that is #1099's fix,
  included because J3b sits on that stack). This PR's own body does not.

## 5. Rose audit

Verdict: **clean for the rebase claim.** The unique work is still the four J3b
SE sites plus the shared harness. The `#1099` ladder is inherited, not
re-implemented. Status stays draft. No `Closes` for a partial #1073 fix.

## 6. CI retrigger

The force-push landed while the PR still targeted `#1099`'s branch, so GitHub Actions (`pull_request` / `branches: [main]`) did not run. Retargeted to `main`; this prose commit is a `synchronize` on the main-based PR so CI can start. The full local suite on `a2df49c12` is unchanged (16,763 / 0).
