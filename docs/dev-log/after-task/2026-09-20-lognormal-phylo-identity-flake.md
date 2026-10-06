# After-task: lognormal phylo/relmat identity flake (PR #785)

## 1. Goal

Root-cause and fix the nondeterministic 1e-10 identity flake in `test/test_lognormal_structured_mean.jl` (PR #781, CI run 35526175941, Julia 1.10 shard 3/4, ubuntu): `theta` off by 5.6e-9 and `re_sd[:species]` by 2.4e-9 between the LogNormal fit and its Gaussian-on-log(y) reference, on a file unchanged since 2c9452dc2.

## 2. Implemented

- Both identity testsets (phylo and relmat) now fit the Gaussian reference on `log.(y)`, the same logged copy `_fit_lognormal_structured` builds, instead of the pre-`exp` simulation truth `logy`.
- The `theta`, `coef(:mu)` and `re_sd` pins changed from `≈ … atol = 1e-10` to exact `==`. No tolerance was widened.
- A comment in the test records the mechanism and the CI numbers.
- Branch `fix/lognormal-phylo-identity-flake` off `origin/main` (5b4b17a98), draft PR #785. PR #781 untouched.

## 3a. Decisions and Rejected Alternatives

- **Chosen: same bits, exact pin.** The delegation is a plain carry-over of theta/vcov/ranef from `drm(f, Gaussian(); data = log.(y), …)`; feeding the reference the same bits makes the identity exact, which is the stronger and more honest statement of the contract. `test_bivariate_lognormal.jl` already did this.
- **Rejected: property bounded by the optimiser tolerance against `logy`.** That would test how far a one-ulp data perturbation moves an LBFGS stopping point, not the delegation. The bound would have to be about `g_tol / λ_min(H)`, on the order of 1e-8, which says nothing about the code under test.
- **Rejected: widen the number.** Forbidden by the brief, and the measured gap (4e-9 here, 5.6e-9 on CI) has no principled ceiling below `g_tol`.
- **Not done: change `_fit_lognormal_structured`.** It is already bit-identical to the Gaussian path on the same data; nothing in `src/` needed to move.

## 4. Files Touched

- `test/test_lognormal_structured_mean.jl` (modified)
- `docs/dev-log/after-task/2026-09-20-lognormal-phylo-identity-flake.md` (this report)

## 5. Checks Run

All local runs on Julia 1.10.12 (CI's failing version) with `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`, the lane convention.

| check | result |
|---|---|
| `log.(exp.(logy)) .== logy` on the test's data | 96 of 240 entries differ on aarch64, 99 of 240 on x86 under Rosetta, max 1.1e-16 |
| max abs diff theta_ln vs theta_g(logy), aarch64, BLAS threads 1 and 4, 3 reps each | 1.8e-15 every run |
| same on x86 Julia 1.10.12 under Rosetta | 4.4e-15 every run |
| theta_ln == theta_g(log.(y)) | true in all 12 configurations |
| 30 random one-ulp perturbation patterns of log(y), Gaussian phylo refit vs unperturbed | min 1.9e-16, median 2.2e-15, max 4.0e-9; 2 of 30 above 1e-10 |
| fixed test file standalone, 3 runs | 17/17 each |
| shard 3/4 via `Pkg.test` at one thread (`--check-bounds=yes`) | passed, 134 testsets, 0 failures — but see below |
| **shard 2/4**, which is where the fixed file actually lands on this branch | passed locally, 154 testsets, 0 failures, `#563 LogNormal structured mean (phylo/relmat)` 17/17 |
| CI `Julia 1.10 - shard 2/4` and `Julia 1 - shard 2/4` | both pass |
| CI on PR #785, all four `Julia 1.10` shards and `Julia 1` shards 1, 2, 4 | pass, 0 failing |

**Shard-number correction.** The brief and the CI failure both name shard 3/4, but that was PR #781's head. On this branch (off `origin/main` 5b4b17a98) `test_lognormal_structured_mean.jl` is file 142 of 269 in the sharded include list, so `(142 - 1) % 4 == 1` puts it in **shard 2**. The first local shard 3/4 run therefore passed without ever loading the fixed file; it is evidence of no collateral damage, not evidence for the fix. The evidence for the fix is CI's `Julia 1.10 - shard 2/4`, green on the ubuntu x86 runner and Julia version that produced the original failure, plus the local shard 2/4 runs.

Shard membership is positional, so any file added to or removed from the include list moves other files between shards. A shard number in a bug report is only meaningful together with the commit it was observed on.

## 6. Tests of the Tests

- The old assertion fails in principle on the same data: `fit_ln.theta == fit_g(logy).theta` is false on every local run (gap 1.8e-15 or 4.4e-15), so the reference really did differ, and the sweep shows the gap reaches 4e-9 for some ulp patterns. That is the failing case the CI runner hit.
- The new `==` pin would fail if the delegation ever stopped copying theta verbatim (for example by re-optimising from the Gaussian solution or re-rounding a parameter), which is exactly the contract under test.
- Both sides of the `==` run in the same process under the same compiler flags, so the 2026-09-19 `Pkg.test --check-bounds=yes` lesson does not apply to it.

## 7a. Issue Ledger

- Fixed: the unsound reference in both testsets of one file.
- Observed, not changed: three `test_joint_missing_*` files still require one Julia thread: `test_joint_missing_frontend.jl:8` and `test_joint_missing_two_frontend.jl:23` as `@test Threads.nthreads() == 1`, and `test_joint_missing_two_predictor.jl:7` still as a module-level `error("wrong thread budget")`. Any multi-threaded `Pkg.test` stops at the first of them; the 2026-09-19 arc (commit 3f13ffbe3) did not remove these. Out of scope here; worth its own small PR.
- Observed, not changed: `fit.iterations` is `-1` for structured Gaussian fits (`_fit_structured_gaussian` never records it), so iteration counts cannot diagnose optimiser divergence on these paths. Recorded on PR #785.

## 8. Consistency Audit

Grep of every test file that calls `LogNormal()` for a Gaussian reference built from a pre-`exp` vector: only this file had the pattern; `test_bivariate_lognormal.jl` already fits its reference on `log.(y1)`, `log.(y2)`. The other lognormal tests do not compare against a Gaussian fit. `src/lognormal.jl` and `src/bivariate_lognormal.jl` documentation states the identity as "Gaussian on the logged response", consistent with the fix.

## 9. What Did Not Go Smoothly

- The CI job log was not retrievable while the run was in progress; the user-supplied numbers stood in for it.
- The x86 Julia under Rosetta did not reproduce the CI magnitude; the ulp-perturbation sweep did, which is the more general probe.
- `closeout.py new` wrote into the vault because the shell cwd resets between calls; moved by hand.
- First shard 3/4 run used `julia_args = ["-t4"]` (the lane convention) and aborted at `test_joint_missing_frontend.jl:8`, whose `Threads.nthreads() == 1` precondition (see the 2026-09-19 LESSONS entry) failed before my testset was reached. Rerun with one thread, as CI does.

## 10. Known Residuals

- The CI runner's exact CPU and BLAS kernel were not identified; the mechanism is confirmed by sweep rather than by replaying that machine.
- Shard 3/4 on Julia 1 (latest) not run locally; CI on PR #785 covers it.

## 11. Team Learning

Memory receipt: loaded the 2026-09-19 LESSONS entry on pins tighter than the estimator's determinacy; applied it in the direction of "same bits, exact" rather than "loosen".

Golden Set: not in scope.

Lesson filed in the vault (`memory/LESSONS.md`, 2026-09-20): an identity test between two fit paths must hand both paths the same bits; `log∘exp` is not the identity in floating point, and a one-ulp data change moves an LBFGS stop by up to `g_tol / λ_min` a few percent of the time. Probe filed in `memory/WHAT-WORKS.md`: the random one-ulp perturbation sweep reproduces "CI-only" optimiser flakes on any machine.

## 12. Cross-Product Coverage

Covers: LogNormal structured markers `phylo` and `relmat` on the mean, univariate, ML path, `g_tol` default, Julia 1.10.12 on aarch64 and x86-under-Rosetta.

Does NOT cover: the bivariate LogNormal delegation (already same-bits, untouched), `animal`/`spatial` (refused by design), REML paths, Julia 1.12/1.13, the CI x86 runner itself.
