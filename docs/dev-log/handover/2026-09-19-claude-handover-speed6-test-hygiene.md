# Session Handoff: speed6 lane, test-suite hygiene (thread-budget guards, duplicated includes, redundant BLAS pins)
Meta: 2026-09-19 17:05 MDT · from Claude (sub-lane of the speed6 arc; orchestrator session "shinichi-82") · context ~35 %

You are Claude, picking up a finished sub-lane of the DRModels.jl speed6 arc. The arc itself (S3/S5/S5d leaves, `LOOP/lanes/speed6-20260919/`) belongs to the orchestrator session and is NOT handed over here. This document covers only the three test-hygiene slices below and the one thing they still owe.

## Critical Context

1. **All three slices are landed on `claude/lane-speed6-20260919`, pushed, and runtime-confirmed.** Full `Pkg.test()` on HEAD `948ee7821` passed at 17:50 (272 files, 500 testsets, 0 fail, 0 error, 1 broken, S5d's). Nothing is owed or carried over; this sub-lane is closed.
2. **One Julia process per lane in this worktree.** On 2026-09-19 a `pkill -f 'Pkg.test()'` from this session killed another session's suite (details in the after-task §9). Before starting any Julia here: `for p in $(pgrep -f julia); do lsof -p $p | awk '$4=="cwd"{print $9}'; done` and wait if any cwd is this worktree. Kill by PID only, never by pattern.
3. `tools/handoff_gate.sh` reports GATE FAIL on `.unlazy/julia-speed-20260919/gates/leaf-S3.md`, `leaf-S5.md`, `leaf-S5d.md`. Those ledgers belong to the orchestrator's arc (this sub-lane had no unlazy leaf); they were deliberately left unmarked. `git` state of this branch is clean and pushed.

## What Was Accomplished

- `3f13ffbe3` **thread-budget guards.** Five `test_joint_missing_*` files required `Threads.nthreads() == 1` (three as a bare module-level `error("wrong thread budget")` that aborted the 14 files after it). Root cause: the 2026-08-30 parity arc's receipt-provenance rule copied into tests as a precondition; nothing on that path is thread-sensitive and CI runs at 1. Julia-thread clause dropped; BLAS==1 kept as a reported `@test`. Full `Pkg.test()` under `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1` reached "Testing DRModels tests passed" for the first time on this lane: 514 testsets, 0 fail, 0 error, 1 broken (S5d's), 38 min.
- `bd500a8b3` **duplicated includes.** `runtests.jl` had a plain `include` block and a `_shard_include` block sharing nine files (both-sides merge from the 2026-09-03 sharding commit, also on `origin/main`). Merged into one sharded block of 11 unique files. New `test/test_runtests_include_list.jl` parses `runtests.jl` as text in every shard and fails, naming paths, on a duplicate, a plain include of a test file, or a missing file. RED (10 plain, 9 dup) then GREEN 276/276.
- `d75f76248` **redundant BLAS pins.** Nine non-restoring `BLAS.set_num_threads(1)` calls removed from the joint_missing family; the `@test BLAS.get_num_threads() == 1` after eight of them kept as a real precondition on the suite pin (`runtests.jl:35`). All twelve joint_missing files 741/741 under the env pin alone; negative control at `OPENBLAS_NUM_THREADS=2` fails exactly the named check.
- `c3a8d82f5`, `8b7eeed94` after-task report (12 sections, validated): `docs/dev-log/after-task/2026-09-19-joint-missing-thread-budget-guards.md`. Three check-log rows in `docs/dev-log/check-log.d/`.
- Vault: lesson appended to `~/shinichi-brain/memory/LESSONS.md` (2026-09-19, receipt rule vs test invariant; kill by PID).

## Current Working State

- Working: `Pkg.test()` completes under the mandated env (measured on `3f13ffbe3`); every joint_missing file passes with the env pin alone (measured on `d75f76248`); the include-list test is green on HEAD.
- In progress: nothing.
- Measured 17:50: full `Pkg.test()` on HEAD `948ee7821` passed; 500 top-level testsets (the nine duplicated files held 15 testsets, plus 1 new), the same one `@test_broken`.

## Key Decisions & Rationale

- Drop the Julia-thread clause rather than run those files under `julia -t1` (option c of the brief): the tested code has no such requirement and no runbook does this. (after-task §3a)
- Keep BLAS==1 as a `@test`, never as `error()`: a module-level `error()` aborts the whole suite instead of recording one failure.
- The include-list guard parses the file as text. A runtime `Set` in `_shard_include` would need the same `error()` anti-pattern and would miss the plain-include form.
- Pins: keep the check, drop the set. Measured aside: with `OPENBLAS_NUM_THREADS` unset, OpenBLAS starts at 1 on this Mac (Julia 1.10.0, aarch64); on a machine where it starts higher, a standalone run of a joint_missing file now fails one named check instead of drifting numerically.
- Related decisions in the vault: D-88 (lane preflight), D-139 (estimate before run; the 38-min suite was estimated 35 to 40), D-116 (default to acting).

## Landing State

`tools/handoff_gate.sh .` (17:00): branch `claude/lane-speed6-20260919` at `8b7eeed94`, 0 ahead / 0 behind `origin`; 1 untracked item (`scratchpad/`, run logs, never commit); GATE FAIL only on the three arc ledgers named in Critical Context 3 (not this sub-lane's).

| Artifact / branch | Committed | Pushed | PR | State |
|---|---|---|---|---|
| `DRModels.jl` `claude/lane-speed6-20260919` `3f13ffbe3` (guards) | y | y | none (the arc's PR is the orchestrator's) | LANDED |
| same branch `bd500a8b3` (includes + include-list test) | y | y | none | LANDED, runtime-confirmed 17:50 (check-log.d `2026-09-19-full-suite-on-head-after-test-hygiene.md`) |
| same branch `d75f76248` (pins) | y | y | none | LANDED, runtime-confirmed 17:50 (same row) |
| same branch `c3a8d82f5`, `8b7eeed94` (after-task) | y | y | none | LANDED |
| this handover (`docs/dev-log/handover/2026-09-19-claude-handover-speed6-test-hygiene.md`) | y (see chat note) | y | none | LANDED |
| `~/shinichi-brain/memory/LESSONS.md` entry (2026-09-19, receipt rule vs test invariant) | y, vault commit `c593c33c`, staged line-precisely so the sibling session's uncommitted entry in the same file was left for its writer (D-60) | n/a (vault is local-only, D-37) | n/a | LANDED |

No `CARRIED-OVER` rows.

## Next Immediate Steps

1. **DONE 2026-09-19 17:50: one full `Pkg.test()` on HEAD** (38 min, log `scratchpad/pkgtest_head.log`, check-log.d row `2026-09-19-full-suite-on-head-after-test-hygiene.md`). Kept for the record; the command was:
   ```bash
   cd /Users/z3437171/local-scratch/lanes/DRM.jl-speed6-20260919 && env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.test()' > scratchpad/pkgtest_head.log 2>&1; tail -3 scratchpad/pkgtest_head.log
   ```
   Result: "Testing DRModels tests passed"; 500 testsets; include-list testset 276/276; the three `thread budget` checks green. Standing note: if a joint_missing file ever fails only its `thread budget` check, the suite pin in `runtests.jl:35` was removed or moved; do not re-add per-file set calls.
2. Nothing. The arc's remaining work (S3/S5/S5d ledgers, PR, merge) is the orchestrator's; read `LOOP/lanes/speed6-20260919/checkpoint.md` before touching anything there, and only if that session has handed it to you.

## Blockers / Open Questions

- None blocking. Unresolved but closed as an observation: the owner of a `Pkg.test()` process (PID 33737) killed at 15:15 was never identified; the BLAS-leak session says it was not theirs. Also observed, not acted on: two of the orchestrator's own Julia process groups ran side by side in this worktree between 16:00 and 16:37.

## Gotchas & Failed Approaches

- Do not "fix" the joint_missing thread guards by running those files under `julia -t1`; the requirement was never real (after-task §2.1).
- Do not put a precondition in a test file as a bare `error()`; the three that did hid 14 files for two weeks.
- Do not add `BLAS.set_num_threads(1)` back into individual test files; the suite pins once and guards per file (`4679936dc`), and the per-file `@test` now checks that pin.
- The Bash tool caps at 10 min even in the background; a full suite needs `nohup` plus a log-tail monitor.
- `tools/check-after-task.R` and `tools/handoff_gate.sh` both halt on the arc's `.unlazy` ledgers from this worktree; that is the arc's state, not a defect in the after-task report.
- The first BLAS-leak theory (task_c0bea3c4) was disproved by measurement before this work started; do not reopen it (`docs/dev-log/after-task/2026-09-19-blas-thread-drift-investigation.md`).

## Mission control

| Repo | Branch / main | CI | What shipped (this sub-lane) | Plan by leverage |
|---|---|---|---|---|
| DRModels.jl | `claude/lane-speed6-20260919` @ `8b7eeed94`, pushed; `origin/main` still carries the nine duplicated includes until the arc merges | not run on this branch by this session (local `Pkg.test()` is the gate per hub rule) | `Pkg.test()` completes under the mandated env; nine files stop running twice; one file stops running in every shard; include-list test; nine dead pins removed | 1) full suite on HEAD (38 min) · 2) nothing further here |

## How to Resume

Environment: worktree `/Users/z3437171/local-scratch/lanes/DRM.jl-speed6-20260919`, branch `claude/lane-speed6-20260919`, Julia 1.10.0 via juliaup, `JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`. Never stage `scratchpad/`. Cheap safe check (3 s, loads only Test): `julia -t1 -e 'include("test/test_runtests_include_list.jl")'`.

Read in this order: repo `AGENTS.md` and `CLAUDE.md`; this document; `docs/dev-log/after-task/2026-09-19-joint-missing-thread-budget-guards.md`; `LOOP/lanes/speed6-20260919/checkpoint.md` (the orchestrator's state, read-only for you). Run `~/shinichi-brain/tools/lane_preflight.sh .` first and claim a lease (`lane_lease.sh --claim DRM.jl-speed6-20260919 --paths docs/dev-log/check-log.d/`) before writing. Classify each item above as OWED / DONE / RETRACTED / PROTECTED against the live git state before acting.

Paste in a fresh Claude session started in the worktree:

```text
Read AGENTS.md and docs/dev-log/handover/2026-09-19-claude-handover-speed6-test-hygiene.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```

> Related: [[2026-09-19-joint-missing-thread-budget-guards]] · [[2026-09-19-blas-thread-drift-investigation]] · `LOOP/lanes/speed6-20260919/checkpoint.md`
