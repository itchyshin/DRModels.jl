## 1. Goal

Find why `test/test_q4_perf_identities.jl`'s `gate_vcov` and `gate_vcov_scaling` fail only inside `Pkg.test()` (suspected: a test file leaving `BLAS.get_num_threads()` at 2), fix it at the root, and add a regression guard that would have caught it.

## 2. Implemented

The premise was wrong, and the measurement says so before any fix was attempted.

1. **No BLAS thread-count leak exists in the suite.** A per-file guard in `test/runtests.jl`'s `_shard_include` (asserts `BLAS.get_num_threads()` equals the suite-start value and `DRModels._blas_pin_scopes[] == 0` after every included file) ran silently through all 474 top-level testsets of a full `Pkg.test()` (JULIA_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1), including after `test_inference_blas_pinning.jl`.
2. **The real cause is `Pkg.test()`'s `--check-bounds=yes` flag.** `julia --check-bounds=yes --project=.` in a pristine session, with no test file included, reproduces the in-suite failure bit-for-bit: `norm(V) = 10.8725488495064` (pinned 10.8720968285066), decay ratio 1.43 (pinned 285). Under the default `--check-bounds=auto` the pinned numbers reproduce bit-exactly (`max|θ̂ − pinned| = 0`). Mechanism, measured at one fixed θ: bounds-check codegen shifts the inner Newton mode within its own tolerance (`marginal_nll` 826.70364502604 vs 826.70364507503, +4.9e-8), the outer LBFGS with `g_tol = 1e-3` then stops 1.9e-5 away in θ̂, and the 1/2h FD Hessian turns that into 4e-5 relative in V. The gates are pinned tighter than the estimator's own determinacy under any codegen change.
3. **Shipped anyway, because it is cheap and the suite needs it:** `test/runtests.jl` now pins `BLAS.set_num_threads(1)` once at suite start (the repo's stated invariant; ten `test_joint_missing_*` files already assume it, and CI, which does not set OPENBLAS_NUM_THREADS, used to run the head of each shard at 2 and the tail at 1) and keeps the per-file guard, so any future leak fails loudly naming the leaking file.

Nothing in `src/` changed. `test/test_q4_perf_identities.jl` is the main lane's file and was not edited; the correction it needs is in §7a.

## 3a. Decisions and Rejected Alternatives

- **Measure before fixing.** The task asked to "fix (either a missing restore, or a genuine race in `_with_pinned_blas`)". Reading `_with_pinned_blas` (lock covers both transitions; restore only at scope count 0) and `test_inference_blas_pinning.jl` (17/17 passed in-suite, which proves no open scope preceded it and that it restores exactly what it observed) showed both were self-consistent, so no fix was applied on suspicion.
- **Guard is start-relative in code but the suite pins 1 first.** Rejected: guard against a literal 1 without pinning (wrong on any platform that starts elsewhere). Rejected: start-relative guard without the pin (would turn CI red at the first `test_joint_missing_*` file, which sets 1 and never restores). The pin changes CI's BLAS count from "2 then 1" to "1 throughout"; flagged in §10 for veto.
- **Did not touch the ten non-restoring `set_num_threads(1)` calls** (`test_joint_missing_*`, and the uncommitted one at the top of the q4 file): with the suite pin they are no-ops everywhere. Removing them is cosmetic and outside this lease.
- **Did not remove the guard once the premise fell.** It costs one `get_num_threads` call per file, and it is the only thing that turns "BLAS drifted" from a downstream numerical mystery into a named file.

## 4. Files Touched

- `test/runtests.jl`: suite-level `BLAS.set_num_threads(1)`; `_check_blas_restored` guard called from `_shard_include`.
- `docs/dev-log/check-log.d/2026-09-19-blas-thread-drift-investigation.md`: gate row.
- `docs/dev-log/after-task/2026-09-19-blas-thread-drift-investigation.md`: this report.

Read but not edited: `src/inference.jl`, `test/test_inference_blas_pinning.jl`, `test/test_q4_perf_identities.jl`, `src/sparse_aug_plsm.jl`, `src/fit_q4_sparse_tmb.jl`.

## 5. Checks Run

All with `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`, Julia 1.10.0, this worktree.

| Check | Result |
| --- | --- |
| Probe inside the real `Pkg.test()` subprocess (`julia_args=-e`): BLAS at start and after all test deps load | 1 and 1 |
| Full `Pkg.test()` with the guard (`scratchpad/guarded_run1.log`, 13:52–14:28) | 474 testsets, guard never fired; q4 4 pass / 3 broken (main lane's `@test_broken`); suite then aborted at `test_joint_missing_uncertainty.jl:3` `error("wrong thread budget")` (pre-existing, see §7a) |
| E2: q4 gates in isolation, pristine vs OpenBLAS pool started (`set(2); set(1)`) | both pass, identical numbers |
| Prefix bisect in the real sandbox: files 1..K then the two gates, K ∈ {40, 20, 5, 0} | all fail with the identical signature (`norm(V)=10.8725488495064`) |
| `julia --check-bounds=yes --project=.`, no test files | fails with the identical signature |
| E4: `marginal_nll` and `fit_q4_sparse_tmb` at fixed θ under `auto` vs `yes` | nll +4.9e-8; θ̂ max diff 1.9e-5 vs 0 under `auto` |
| `Meta.parseall(runtests.jl)` | parses |
| `DRM_TEST_SHARD=1/30` and `2/30` through `Pkg.test()` with the final `runtests.jl` | both pass: 1/30 (9 files incl. q4, 30 testsets) and 2/30 (9 files, 34 testsets), guard silent, exit 0 |

## 6. Tests of the Tests

The guard was exercised with injected faults in a plain session: a clean file emits nothing; `BLAS.set_num_threads(2)` before the check emits `BLAS state restored after leaky_file.jl | 1 1 2` and throws `TestSetException`; `_blas_pin_scopes[] = 1` emits the `scopes == 0` failure and throws. So the negative control fails and names the file, and the positive control is silent.

The bisect was validated by its own negative space: the K=0 prefix (no test file at all) already fails, and a plain `--check-bounds=yes` session with no `runtests.jl` involvement fails identically, so no test file can be the cause.

## 7a. Issue Ledger

Fixed here:
- Suite-level BLAS pin + per-file drift guard (`test/runtests.jl`).

For the main lane (owner of `test/test_q4_perf_identities.jl`), not fixed here:
- The file's header comment (line ~30) and the closing testset comment attribute the in-suite failure to `test_inference_blas_pinning.jl` leaving BLAS at 2, then to a lingering `Threads.@spawn` task. Both are measured false. The two `@test_broken` gates should instead be re-pinned under `Pkg.test()` conditions (`julia --check-bounds=yes`) or bounded by what the fit actually determines (`g_tol = 1e-3` does not fix θ̂ to rtol 1e-6, and an FD Hessian at h=1e-4 does not fix V to rtol 1e-8 under any codegen change). The file-top `BLAS.set_num_threads(1)` is now redundant.

Pre-existing, reported not touched:
- `test/test_joint_missing_uncertainty.jl:3`, `test_joint_missing_bridge.jl:3`, `test_joint_missing_two_predictor.jl:7` (and siblings) `error("wrong thread budget")` at module level unless `Threads.nthreads() == 1`, so `Pkg.test()` can never complete under the documented `JULIA_NUM_THREADS=4`. On `origin/main` since the rename commit `90fbb0e28`.
- `test/runtests.jl:294–313` includes the same nine files twice (plain `include` then `_shard_include`), on `origin/main`; exactly the merge-duplication the file's own maintenance note warns about.

## 8. Consistency Audit

- `_with_pinned_blas` (`src/inference.jl:117`) and its three call sites (`profile_result`, `_ls_profile_result`, `_bootstrap_result`) all fetch spawned tasks inside the scope; no leak path. `_profile_row_result` leaves the `right` task running if `fetch(left)` throws (a stray task, not a BLAS-state leak).
- Every `set_num_threads` call in `test/` and `src/` enumerated: only the pinning test sets a value other than 1, and it restores.
- No test file monkeypatches an existing DRModels method (the `DRModels._ls_*` definitions in `test_locscale_inner_status.jl` add methods on test-local types only); no q4 helper or `const` name collides with another test file.
- The sandbox's dependency versions equal the main `Manifest.toml` (Optim 1.13.3, LineSearches 7.5.1, OpenBLAS_jll 0.3.23+2, SuiteSparse_jll 7.2.1+1).
- CI (`.github/workflows/CI.yml`) sets neither `OPENBLAS_NUM_THREADS` nor `JULIA_NUM_THREADS`; the new pin makes shards uniform.

## 9. What Did Not Go Smoothly

- The task statement asserted the mechanism ("leaves BLAS at 2 … MEASURED"). It was an inference from a signature match; two 25-minute full runs were spent by the previous session on a workaround (`set(1)` at file top) that could not work because the count was already 1.
- A second `Pkg.test()` from the previous session was still running in this worktree when this one started, and the main lane edited the q4 file at 14:07 mid-run; both were identified by process cwd and file mtime rather than assumed. Lease narrowed to exclude the q4 file.
- First prefix-bisect launch resolved `include(f)` relative to the scratch script; fixed and relaunched.
- The Test stdlib block-buffers stdout when redirected, so the 35-minute guarded run could not be watched mid-flight.

## 10. Known Residuals

- **CI behaviour change to confirm:** with the pin, every CI shard runs at BLAS=1 throughout. Previously the head of each shard ran at 2. Numerically this is the invariant the pins assume; timing effect expected negligible (small dense problems). Veto by deleting the one `BLAS.set_num_threads(1)` line; the guard then needs the joint-missing files to restore.
- `Pkg.test()` still cannot complete under `JULIA_NUM_THREADS=4` because of the joint-missing thread-budget guards (§7a); that is a decision for their author.
- The q4 gates remain `@test_broken` until the main lane re-pins them.

## 11. Team Learning

Memory receipt: routed guards loaded: hub `AGENTS.md` (lane preflight, D-88; estimate-before-run, D-139; default-to-acting, D-116), repo `AGENTS.md`/`CLAUDE.md` (TDD, check-log, Rose audit), `systematic-debugging` (root cause before fix; this is the one that shaped the work, since it forbade patching `_with_pinned_blas` on suspicion). Lane preflight run; lease claimed and narrowed. Golden Set: not in scope (no known-mistake class matched).

Durable lesson (filed to the vault LESSONS): `Pkg.test()` runs the suite with `--check-bounds=yes`; any numerical pin measured in a plain REPL is measured under different codegen. Pin under `julia --check-bounds=yes`, or bound by the estimator's own determinacy. A signature match ("set(2) reproduces it") is not a mechanism; K=0 of a prefix bisect is the cheapest disproof.

## 12. Cross-Product Coverage

Covers: the q=4 bivariate phylo ML route's vcov gates; `Pkg.test()` on macOS aarch64, Julia 1.10.0; JULIA_NUM_THREADS=4; the suite's BLAS-count invariant on every platform.

Does NOT cover: the `'1'` CI leg's Julia version, where the same `--check-bounds=yes` effect is expected but not measured; Linux CI runners; whether the q4 pins reproduce under `--check-bounds=yes` across Julia versions (they may not, which is the argument for bounding by determinacy rather than re-pinning).
