# After-task: remove the one-thread preconditions from the joint-missing tests

## 1. Goal

Remove the leftover `Threads.nthreads() == 1` preconditions from the `test_joint_missing_*` files so the suite runs under any thread count. Under `DRM_TEST_SHARD=3/4` with four Julia threads the shard stopped at `test_joint_missing_frontend.jl:8` (44 passed, 1 failed) and every later file in the shard went unrun.

## 2. Implemented

Five files, all in the same class. The brief named three; a sweep found two more.

| file | before | after |
|---|---|---|
| `test/test_joint_missing_frontend.jl:8` | `@test Threads.nthreads() == 1` | removed |
| `test/test_joint_missing_two_frontend.jl:23` | `@test Threads.nthreads() == 1` | removed |
| `test/test_joint_missing_two_predictor.jl:7` | module-level `… \|\| error("wrong thread budget")` | removed, comment explains why |
| `test/test_joint_missing_uncertainty.jl:3` | module-level `… \|\| error("wrong thread budget")` | removed, comment |
| `test/test_joint_missing_bridge.jl:3` | module-level `… \|\| error("wrong thread budget")` | removed, comment |

Every file keeps its own `BLAS.set_num_threads(1)` pin, and the two frontend files keep `@test BLAS.get_num_threads() == 1`. BLAS is self-enforced by the line above it, so that assertion can always pass; the Julia thread count could not be.

## 3a. Decisions and Rejected Alternatives

- **Chosen: delete the thread-count clause, keep the BLAS pin.** The 2026-09-19 lesson is that this was the parity arc's *receipt* rule ("evidence produced at one Julia and one BLAS thread") copied into the tests as a precondition. The tests are not thread-sensitive, so the clause only ever encoded provenance.
- **Rejected: move the clause into a `@testset` so it fails softly.** It would still fail under the lane's mandated four threads, for a property nothing in the file depends on. A precondition nothing needs is not worth a soft failure either.
- **Rejected: change the `tools/check_*.jl` scripts.** Eight of them carry the same clause. Those scripts *generate* the receipts, so the provenance rule belongs there, and they are run deliberately, not by the shard. Left untouched on purpose.
- **Left alone: `test/test_bootstrap_thread_flags.jl:16`.** Its `Threads.nthreads()==1 || minimum(counts)==B` is genuinely thread-adaptive, asserting behaviour as a function of the thread count rather than demanding one.

## 4. Files Touched

- `test/test_joint_missing_frontend.jl` (modified)
- `test/test_joint_missing_two_frontend.jl` (modified)
- `test/test_joint_missing_two_predictor.jl` (modified)
- `test/test_joint_missing_uncertainty.jl` (modified)
- `test/test_joint_missing_bridge.jl` (modified)
- `docs/dev-log/after-task/2026-09-20-joint-missing-thread-preconditions.md` (this report)

## 5. Checks Run

All runs on Julia 1.10.12, `OPENBLAS_NUM_THREADS=1`.

**Standalone, each file at both thread counts:**

| file | 1 thread | 4 threads |
|---|---|---|
| `test_joint_missing_frontend` | pass, 56 assertions | pass, 56 |
| `test_joint_missing_two_frontend` | pass, 61 | pass, 61 |
| `test_joint_missing_two_predictor` | pass, 83 | pass, 83 |
| `test_joint_missing_uncertainty` | pass, 19 | pass, 19 |
| `test_joint_missing_bridge` | pass, 42 | pass, 42 |

Identical assertion counts at both thread counts, which is what you see only if nothing in these paths is thread-sensitive.

**Under `Pkg.test` (`--check-bounds=yes`), four Julia threads**, the configuration that used to abort:

| shard | holds | result |
|---|---|---|
| 3/4 | `test_joint_missing_frontend.jl` | pass, 134 testsets, 0 failures (previously: 44 passed, 1 failed, rest of shard unrun) |
| 4/4 | `two_predictor`, `two_frontend`, `bridge` | pass, 135 testsets, 0 failures |
| 2/4 | `uncertainty` | pass, 154 testsets, 0 failures |

Shard 3/4 also rerun at **one** thread: pass, 134 testsets, 0 failures, so CI's configuration is unaffected.

The `joint missing-predictor formula frontend` testset now reports **44 of 44**. Before the change, at four threads it reported 44 passed and 1 failed out of 45. Exactly the bogus assertion is gone and every real one still passes.

**Shard membership.** The five files span three shards, not one: indices 256, 258, 259, 260 and 264 of 269 give shards 4, 2, 3, 4 and 4. So the three module-level `error(...)` calls were three independent abort points in three different shards, not one. Verifying the change needs all three shards, and a shard number quoted in a bug report only means something together with the commit it was seen on.

## 6. Tests of the Tests

- The failure the change removes is directly demonstrated: on `origin/main`, shard 3/4 at four threads fails at `test_joint_missing_frontend.jl:8` and abandons the rest of the shard. Evidence in the PR #785 lane log, same machine and Julia version.
- The change cannot mask a real defect: each file is run at one thread and at four, and every assertion in them must still pass at both. If anything in the joint-missing paths were thread-sensitive, the four-thread run would show it.
- Verified there is nothing to be sensitive to: no `@threads`, `@spawn`, thread-count kwarg, or RNG in the five files, and `grep` over `src/joint_missing*.jl` finds no threading at all.

## 7a. Issue Ledger

- Fixed: five test-side thread-count preconditions, the whole test-side class.
- Deferred by design: eight `tools/check_*.jl` receipt scripts keep the clause (see decisions).

## 8. Consistency Audit

Swept `test/` and `tools/` with `grep -rnE "nthreads\(\) *== *1|wrong thread budget"`, a pattern that matches both spacing conventions. The first, narrower sweep missed two sites because the files omit spaces around `==`; the wider pattern found the full set of fifteen, which partitions into five test-side preconditions (fixed), one thread-adaptive assertion (correct), and nine tool-script receipt guards (correct where they are).

## 9. What Did Not Go Smoothly

- The first sweep used `nthreads() == 1` with spaces and excluded the `test_joint_missing` prefix, so it reported the class clear twice while two sites remained. Fixed by matching both spacings across the whole of `test/` and `tools/` in one pass. The lesson is to write the pattern for the sloppiest spelling in the repo, not the one in front of you.
- This lane and a spawned background task were both pointed at the same work; this lane holds the lease.

## 10. Known Residuals

- The four-thread shard was run on aarch64 Julia 1.10.12 only. CI runs one thread, so CI itself never exercised the removed clause.
- `tools/check_*.jl` receipt scripts were not run; they are out of scope and unchanged.

## 11. Team Learning

Memory receipt: loaded the 2026-09-19 LESSONS entry on receipt-provenance rules leaking into tests, and its companion rule that a bare module-level `error()` in a test file is a suite abort rather than a test failure. Both applied directly: this change is that lesson's unfinished half.

Golden Set: not in scope.

Durable lesson for the vault: when a cleanup arc removes a bad pattern, grep for it with the sloppiest spelling the repo uses, or the arc leaves survivors. The 2026-09-19 arc fixed the sites it saw and left five, three of which differ only by whitespace around `==`.

## 12. Cross-Product Coverage

Covers: the `test_joint_missing_*` family under Julia thread counts 1 and 4, BLAS pinned to one thread, Julia 1.10.12 on aarch64, shard 3/4 and standalone.

Does NOT cover: Julia 1.12/1.13, x86 runners, BLAS thread counts above one, the `tools/check_*.jl` receipt scripts, and thread counts other than 1 and 4.
