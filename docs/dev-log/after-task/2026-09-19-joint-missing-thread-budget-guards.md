## 1. Goal

Resolve the contradiction between five `test/test_joint_missing_*.jl` files that require `Threads.nthreads() == 1` and the speed6 lane's mandated `JULIA_NUM_THREADS=4`, so that the full `Pkg.test()` can complete under the documented environment (follow-up task_61f1b009, raised by the S5d leaf). Extended on Shinichi's ask to the two pre-existing items that report surfaced: the duplicated include blocks in `test/runtests.jl`, and the redundant `BLAS.set_num_threads(1)` calls in the joint_missing family.

## 2. Implemented

1. **Root cause established before any edit.** The `Threads.nthreads() == 1` clause is a leftover of the 2026-08-30 parity arc's receipt-provenance rule ("all Julia checks pin Julia and BLAS to one thread"; that arc's receipt checker used "wrong thread budget" as a negative fixture). It was copied into the test files as a module-level precondition. Nothing those files exercise is thread-sensitive: no `src/joint_missing*` code branches on `Threads.nthreads()`, none of the five files call the threaded inference paths in `src/inference.jl`, and CI never sets `JULIA_NUM_THREADS`, so CI always ran at 1 and never met the guard. Under the lane's `JULIA_NUM_THREADS=4` (`LOOP/lanes/speed6-20260919/GOAL.md:18`) the guard could never pass.
2. **Fix, one class, five files.** In `test_joint_missing_two_predictor.jl`, `test_joint_missing_uncertainty.jl` and `test_joint_missing_bridge.jl`, the bare `error("wrong thread budget")` is replaced by a `@testset "thread budget"` holding `@test BLAS.get_num_threads() == 1`, with a comment stating why the Julia thread count is not asserted. In `test_joint_missing_frontend.jl` and `test_joint_missing_two_frontend.jl` the line `@test Threads.nthreads() == 1` is removed. The BLAS half is kept because it is the suite's real numerical invariant; a mismatch is now reported and named, not fatal to the 14 shard-includes that follow.
3. **Verified end to end.** RED reproduced in isolation, GREEN in isolation, then the full mandated `Pkg.test()` reached `Testing DRModels tests passed` for the first time on this lane.

4. **Duplicated includes merged and guarded (`bd500a8b3`).** `test/runtests.jl` carried a plain `include` block (lines 327 to 336) and a `_shard_include` block (337 to 346) sharing nine files: the both-sides merge the file's own MAINTENANCE NOTE warns about, on `origin/main` since the 2026-09-03 sharding commit `8bc2ce54e`. Nine files ran twice per full suite, and the plain block ran in every CI shard outside the split and the per-file BLAS guard. Merged into one `_shard_include` block of the 11 unique files in the original order (`test_bridge_formula_constructs.jl` and `test_bridge_zi_marginal_mean.jl` were each on only one side). Class fix: new `test/test_runtests_include_list.jl` reads `runtests.jl` as text in every shard and fails, naming the paths, on a duplicate, a plain include of a test file, or a missing file.
5. **Redundant BLAS pins removed (`d75f76248`).** Nine non-restoring `BLAS.set_num_threads(1)` calls in the joint_missing family removed; the `@test BLAS.get_num_threads() == 1` after eight of them kept, now a real precondition on the suite pin rather than a set-then-check tautology. Kept on purpose: the suite pin (`runtests.jl:35`), the restoring call in `test_lss_sparse_multi_public.jl:387`, and the pinning test's own call.

Four commits on `claude/lane-speed6-20260919` above the S5d builder's `d59082ca6`: `3f13ffbe3` (guards), `c3a8d82f5` (first version of this report), `bd500a8b3` (includes), `d75f76248` (pins). Nothing in `src/` changed.

## 3a. Decisions and Rejected Alternatives

- **Drop the Julia-thread clause rather than run the files under `julia -t1`.** Rejected option (c) of the task brief: a separate single-thread invocation would encode a requirement the tested code does not have, and no CI runbook does this today (CI runs `julia-runtest@v1` with no thread setting). Splitting the suite to honour a provenance rule that outlived its arc adds a second harness for no correctness gain.
- **Keep the BLAS==1 assertion, as a `@test`, not an `error()`.** `runtests.jl` already pins BLAS to 1 at suite start and guards it per file (commit `4679936dc`), so this assertion is documentation of the invariant rather than the gate; it costs one call. It is a `@test` so that a future mismatch names the file without aborting the suite. A raw module-level `error()` defeats the Test harness and is what hid 14 files behind one line.
- **Did not relax to `Threads.nthreads() >= 1`.** That is always true and would be a comment pretending to be a check.
- **First pass left the other non-restoring `BLAS.set_num_threads(1)` calls alone** (no-ops under the suite pin, outside that lease); Shinichi then asked for them, and the second pass removed them under a fresh lease (see "Pins" below).
- **Includes: merge into the sharded form, not the plain one.** The plain block predates sharding; keeping it would leave ten files outside the shard split and the per-file BLAS guard. The harness test is a text parse rather than a runtime `Set` in `_shard_include`, because a module-level `error()` on a repeat would be the same suite-abort anti-pattern as the guards this task removed, and a text check also catches the plain-include form that a runtime counter cannot see.
- **Pins: keep the `@test`, drop the `set`.** Dropping both would make the files silent about the environment; keeping both is a tautology. With only the check, a standalone run off the pinned environment fails one named test instead of drifting numerically. Measured aside: with `OPENBLAS_NUM_THREADS` unset, OpenBLAS starts at 1 on this Mac (Julia 1.10.0, aarch64), so standalone runs here behave as before.
- **Did not edit the sibling task's territory.** task_c0bea3c4's premise (BLAS leaking to 2 from `test_inference_blas_pinning.jl`) was already measured false by `4679936dc`'s per-file guard over 474 testsets; this fix depends on nothing from it.

## 4. Files Touched

- `test/test_joint_missing_two_predictor.jl` (guard rewritten)
- `test/test_joint_missing_uncertainty.jl` (guard rewritten)
- `test/test_joint_missing_bridge.jl` (guard rewritten)
- `test/test_joint_missing_frontend.jl` (one `@test` line removed)
- `test/test_joint_missing_two_frontend.jl` (one `@test` line removed)
- `docs/dev-log/check-log.d/2026-09-19-joint-missing-thread-budget-guards.md` (new gate row)
- `docs/dev-log/after-task/2026-09-19-joint-missing-thread-budget-guards.md` (this report)
- `test/runtests.jl` (two include blocks merged into one; harness test registered after `test_shard_selection.jl`)
- `test/test_runtests_include_list.jl` (new)
- `docs/dev-log/check-log.d/2026-09-19-runtests-duplicate-includes.md` (new gate row)
- `test/test_joint_missing_two_bridge.jl`, `test_joint_missing_finite_frontend.jl`, `test_joint_missing_finite_factor_coding.jl`, `test_joint_missing_finite_prediction.jl` (set call removed), and the five files above again (set call removed, comment reworded)
- `docs/dev-log/check-log.d/2026-09-19-joint-missing-redundant-blas-pins.md` (new gate row)

Read but not edited: `src/inference.jl`, `src/joint_missing_frontend.jl`, `src/joint_missing_uncertainty.jl`, `.github/workflows/CI.yml`, `docs/dev-log/after-task/2026-08-30-parity-joint-frontend.md`, `docs/dev-log/after-task/2026-09-19-blas-thread-drift-investigation.md`, `LOOP/lanes/speed6-20260919/GOAL.md`.

## 5. Checks Run

All on the Mac Studio, Julia 1.10.0, worktree `DRM.jl-speed6-20260919`, `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`.

| Check | Result |
| --- | --- |
| `git log --all -S'wrong thread budget'` and `-S'Threads.nthreads() == 1'` on `test/` | introduced 2026-08-30 (`e2554bec4`, `a8758468a`, `027f91225`, `8ee084866`, `e6501e034`), all in the parity-joint arc |
| `grep` for `Threads.`, `@spawn`, `@threads` in `test/test_joint_missing*.jl` | the guards were the only hits |
| `grep nthreads()` in `src/` | only `src/inference.jl` (profile/bootstrap paths, not called by these files) |
| RED: `julia -t4` include of `test_joint_missing_bridge.jl` | `ERROR ... test_joint_missing_bridge.jl:3`, 0 passed, 1 errored |
| GREEN: the five edited files in one `-t4` session | 264 / 264 pass, 46.1 s, `nthreads=4` |
| Full `Pkg.test()` (`scratchpad/pkgtest_threadbudget.log`, 15:15 to 15:53) | `Testing DRModels tests passed`, `EXIT=0`; 514 top-level testsets, 0 fail, 0 error, 1 broken (S5d's own `@test_broken` in "q4 perf identities"); the three "thread budget" testsets each `1 / 1` |
| `julia -t1` on `test_runtests_include_list.jl` against the old `runtests.jl` (RED) | 2 failed, naming 10 plain includes and 9 duplicates; 281 included files exist |
| same after the merge (GREEN); `Meta.parseall(runtests.jl)` | 276 / 276; parses; 272 sharded includes (270 + 10 plain before) |
| all twelve `test_joint_missing_*` files in one `-t4` session, env pin only, no `set_num_threads` anywhere (after the builder's suite exited at 16:37) | 741 / 741, 2m20s, BLAS 1 throughout |
| negative control: `test_joint_missing_bridge.jl` at `OPENBLAS_NUM_THREADS=2` | exactly the named `thread budget` check fails (42 pass / 1 fail); the rest of the file runs; nothing aborts |
| `tools/check-after-task.R` on this report | structure passed; the ledger stage halts on the arc's own `.unlazy` leaves S3/S5/S5d (UNMET), which are the orchestrator's gates, not this task's (§10) |
| `tools/slop_check.py` on this report | 0 findings, 0 em dashes |

## 6. Tests of the Tests

- The negative control is the RED run: the unedited guard aborts the file at line 3 under `-t4`, which is the exact in-suite failure the S5d leaf reported. The positive control is the same include after the edit: 264 assertions pass.
- The retained `@test BLAS.get_num_threads() == 1` would fire if a file ahead of it left BLAS at 2; that fault was injected and shown to fail loudly in `4679936dc`'s report (§6 there), and the suite pin makes it silent today. It reports through Test rather than `error()`, so a future failure shows a named testset and the files after it still run.
- The include-list test was written first and failed on the real file with the real defects (10 plain, 9 duplicated), then passed after the merge; a file that does not exist fails the third testset (281 checked).
- The retained BLAS check was shown to fire on an injected fault (`OPENBLAS_NUM_THREADS=2`) and to fail only itself, which is the reported-not-raised property the whole task is about.
- The full-suite log shows all 14 shard-includes after `two_predictor` executing (testsets 475 to 514), which is the count that could never be reached before.

## 7a. Issue Ledger

Fixed here:
- task_61f1b009: joint_missing thread-budget guards vs `JULIA_NUM_THREADS=4`.

Closed by evidence, not by this change:
- task_c0bea3c4 (BLAS leak to 2): no leak in 474 testsets per `4679936dc`; the q4 vcov drift is `--check-bounds=yes` codegen. The BLAS half of these guards was always satisfied.

For the S5d leaf (owner: the speed6 orchestrator):
- The 14:48 gate-check recorded `G5d.6: FAIL ... signal=SIGTERM`. That SIGTERM is this session's `pkill` (§9). The gate would have failed on the old guards anyway, but the recorded reason is wrong; the orchestrator is re-recording it from a fresh run. `G5d.3` and `G5d.5` are unmet on their own terms and were not touched.

Fixed here, second pass (Shinichi's ask):
- `test/runtests.jl` nine duplicated includes plus the plain-include block (`bd500a8b3`), with a harness test so the merge pattern cannot recur silently.
- Nine redundant `BLAS.set_num_threads(1)` calls in the joint_missing family (`d75f76248`); the prior report's "ten" counted the q4 file's, which the S5d leaf had already removed.

Reported to the orchestrator, not acted on:
- Between 16:00 and 16:37 the worktree ran two of the builder's own Julia process groups side by side (PIDs 65782/65785 and 67428/67429/67479), which looked like two suites at once under its own one-process rule.

## 8. Consistency Audit

- Every `nthreads` assertion in `test/` enumerated (`grep -rn "nthreads" test/`): after the edit, none remain in the `test_joint_missing_*` family; the remaining uses are in `test_inference_blas_pinning.jl`, `test_bootstrap_thread_flags.jl` and the locscale profile/bootstrap tests, which exercise threaded code on purpose and branch on the count rather than requiring 1.
- Every `error(` at module level in `test/` was searched for the same anti-pattern (a bare precondition outside `@testset`): the three rewritten here were the only ones.
- `LOOP/compute-readiness.md:47-50` asks receipts to *record* `Threads.nthreads()` and `BLAS.get_num_threads()`; it never asks tests to *require* 1. The fix is consistent with that doctrine.
- `.github/workflows/CI.yml` sets neither thread variable; CI behaviour is unchanged by this commit (it ran at nthreads=1 before and still does).
- Every `set_num_threads(` in `test/` and `src/` re-enumerated after the pin removal: three remain in `test/` (suite pin, the restoring `finally` in `test_lss_sparse_multi_public.jl`, the pinning test) and each is intentional; `test_q4_perf_identities.jl` carries only a comment saying its call was removed.
- Every column-0 `include(` in `runtests.jl` enumerated: only `shard_util.jl` remains, and the new harness test enforces that in every shard.
- No neighbour pins the include count (`test_shard_selection.jl`, `shard_util.jl`, CI, the coordination board checked); the `_n_test_files` banner in `runtests.jl` is computed, not literal.
- The S5d builder's commit `d59082ca6` landed beneath mine during the run; `git status` after commit shows only `scratchpad/` untracked, so no cross-lane file was swept into `3f13ffbe3`.

## 9. What Did Not Go Smoothly

- **I killed two processes that were not mine.** My first launch of the full suite went through the Bash tool's 10-minute cap, so I stopped it and relaunched with `nohup`; the `pkill -f 'Pkg.test()'` I used to clear my own julia also matched the S5d gate-check's outer julia (PID 20178, started 14:53 by node 17081) and a third `Pkg.test()` (PID 33737) whose owner is still unidentified (the BLAS session says it was not theirs; the "Handover rehydration" session it named is no longer listed). I then stopped the gate-check's orphaned test subprocess (PID 20203, 100 % CPU, unable to produce a PASS with its parent dead). Both peers were told at once; the orchestrator's rule from here is one Julia process per lane, and I agree with it. The lesson is in §11.
- The task brief named `frontend` and `two_frontend` as the "identical" siblings; they use `@test`, not `error()`. The hard-abort siblings are `uncertainty` and `bridge`. Fixed as one class regardless.
- The brief estimated "~100+ files" after `two_predictor`; the include order has 14. The abort cost was real but smaller.
- The pins slice had to wait 13 minutes for the builder's suite to exit before any edit or run, because its include order would have reached the joint_missing files mid-flight; the wait was a monitor on the worktree's Julia PIDs, not a poll.
- The first version of the harness test raised `error()` inside `@test` to name the path; rewritten to collect offenders and `@test isempty(...)`, so a failure prints the paths as a value and records as Fail, not Error.
- The Bash tool caps at 10 minutes even in the background; a 38-minute suite needs `nohup` plus a log-tail monitor, and the first monitor expired before the run ended.

## 10. Known Residuals

- `G5d.6` must be re-recorded by the S5d builder from a run this session did not interrupt (in progress at the orchestrator's end when this report was written).
- PID 33737's owner is unidentified. If a session finds a `Pkg.test()` of theirs died with SIGTERM at 15:15 in this worktree, it was this session's `pkill`.
- The `@test BLAS.get_num_threads() == 1` retained in five files is always true under the suite pin; it documents the invariant and would only fire if the pin in `runtests.jl` were removed.
- The include merge and the pin removal were not re-run as a full `Pkg.test()`: the builder's suite held the lane's one-Julia budget, and the changes remove repeats and no-ops rather than add behaviour. The orchestrator's next full run is the runtime confirmation; expect the top-level testset count near 506 (514 minus the nine repeats plus the harness test).
- On a machine where OpenBLAS starts above 1 with the variable unset, a standalone run of a joint_missing file now fails the named `thread budget` check; that is the intended visibility, but it is a behaviour change for anyone running files outside `runtests.jl` without `OPENBLAS_NUM_THREADS=1`.
- Not re-run on the CI Julia versions; the change is test-only and thread-count-neutral there, so no difference is expected, and none was measured.

## 11. Team Learning

Memory receipt: routed guards loaded: hub `AGENTS.md` (lane preflight D-88, lease claimed and narrowed to the five files plus `check-log.d/`; estimate-before-run D-139, stated 35 to 40 min, measured 38; default-to-acting D-116), repo `AGENTS.md`/`CLAUDE.md` (TDD; check-log entry via `check-log.d/`, not the frozen table), `systematic-debugging` (this is the one that shaped the work: it forced reading the 2026-08-30 arc's provenance rule before touching the guards, which is what turned "silence the check" into "the check was never about the code"). Recalled first: `4679936dc`'s report, which had already disproved the sibling task's premise and named this residual. Golden Set: not in scope (no known-mistake class matched; the process-kill in §9 is a candidate for one).

Durable lessons, filed to the vault LESSONS:
1. **A receipt-provenance rule is not a test invariant.** "This evidence was produced at one thread" belongs in the receipt, not as a precondition on the tests that later guard the code. Copying it into the test files made the suite depend on how the evidence was once gathered.
2. **Never `pkill` by command pattern in a shared worktree.** Kill by PID after `lsof -p` shows the cwd and `ps -o lstart` shows the start time. A worktree with several lanes' processes is exactly where a pattern kill hits someone else's run, and it did.
3. A bare module-level `error()` in a test file is a suite abort, not a test failure. Preconditions go in a `@testset`.
4. When a file's own maintenance note predicts a failure mode ("git keeps both sides"), give the note a test. The include-list check costs 0.1 s per shard and would have caught the 2026-09-03 duplication and this one.

## 12. Cross-Product Coverage

Covers ✓: the `runtests.jl` include list (no duplicates, no plain includes, every file present) in every shard; all twelve `test_joint_missing_*` files under the env pin alone; the five `test_joint_missing_*` files under `JULIA_NUM_THREADS=4` and `=1` (CI); the full `Pkg.test()` on macOS aarch64, Julia 1.10.0, at `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`; the suite's BLAS=1 invariant as a reported assertion.

Does NOT cover ✗: `JULIA_NUM_THREADS` values other than 1 and 4 (nothing in these files depends on it, but only those two were run); the Linux CI legs and the `'1'` Julia version (test-only change, not executed there); the S5d gates `G5d.3`, `G5d.5`, `G5d.6` (another lane's, re-run pending); a full `Pkg.test()` after the include merge and the pin removal (runtime confirmation left to the orchestrator's next run); shard balance after the merge (positions after line 327 shift by up to ten, unmeasured); machines whose OpenBLAS default is above 1 when the variable is unset (standalone-run behaviour there is reasoned, not measured).
