## 1. Goal

Reconcile current PR, CI, and lane-owner evidence for the DRM parity work without changing another lane's source or overstating parity.

## 2. Implemented

Updated the rolling crosswalk with the 23:00 UTC PR refresh, the 23:10 UTC owner-safe follow-up, and a 23:12 UTC terminal check for drmTMB #1657. Its exact-head CI is now fully green, but GitHub still reports it non-mergeable; no formal review or exact-head checkout was found. The update also records exact-head failures for DRModels.jl #1248, a still-running shard for #1249, and the absence of an easy candidate with both a live exact-head checkout and no overlapping owner lane. The earlier refresh records the merges of drmTMB #1663 and DRModels.jl #1242, #1239, and #1234, including their validation limits and the still-dirty SD worktree beside #1663.

The 23:18 UTC correction records that drmTMB main advanced to `e5cd866` after #1663 merged. It also updates #1657: the exact-head run is green, but the PR is now 47 commits behind main, divergent, and reported dirty/non-mergeable. #1249 still has shard 3 running. The latest observed R-CMD-check failure is from October 26 at an older main SHA; its macOS job log has expired, and no current-main R-CMD-check was found.

## 3a. Decisions and Rejected Alternatives

- Kept the older 22:50 PR census as a dated snapshot and added a later state note instead of rewriting historical evidence.
- Did not edit `docs/design/capability-status.md`. A separate detached worktree has uncommitted changes to that page and untracked reports; its owner is not identified in the coordination board or live lease list.
- Did not touch the dirty `general-sd-regression` worktree. Its active-lane ledger assigns cross-engine SD work to Shinichi.
- Did not use Totoro or DRAC because this slice used GitHub metadata, source text, and worktree inspection only.
- Left the green but unowned or overlapping PR branches untouched. No candidate met both the exact-head checkout and collision-free owner conditions.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/evidence/2026-11-02-open-pr-census/README.md`
- `docs/dev-log/after-task/2026-11-02-parity-crosswalk-reconciliation.md`
- `docs/dev-log/check-log.d/2026-11-02-parity-crosswalk-reconciliation.md`

## 5. Checks Run

- Read-only GitHub checks refreshed PR metadata and exact-head workflow jobs for the listed candidates.
- Local worktree status and the active-lane record were read for the merged #1663 worktree.
- Read-only diff review inspected the detached capability-page worktree; no file there was changed.
- Refreshed exact-head PR metadata and run/job results for #1657, #1249, and #1248; inspected logs for all three failing #1248 Julia shards; confirmed all five jobs in #1657 run 41522911628 are now successful.
- Read-only R and Julia lane refreshes found no safe, straightforward PR branch to take; no owner was contacted.
- `git diff --check`: passed.
- `check-after-task.R` on this report: passed after the 23:18 continuation.
- `slop_check.py` on this report: 0 findings across 1,211 words; no em dashes.
- Refreshed the live drmTMB main ref, #1657 compare/API/reviews, #1249 shard status, and the latest 50 main workflow runs. The historical R-CMD-check job log returned HTTP 410; no test-level cause was inferred.
- No package tests, fits, simulation campaigns, or compute jobs were run. Totoro and DRAC are available for later validation work.
- The Julia executable is unavailable in this environment, so `tools/build_check_log.jl --check` could not run. A Python-equivalent format scan found existing narrative-style entries in multiple older `check-log.d/` files; the edited row itself passes the required one-row Markdown shape. Those historical files were left untouched.

## 6. Tests of the Tests

Not run. No source or test code changed.

## 7a. Issue Ledger

The live PR search returns 34 open drmTMB PRs and 24 open DRModels.jl PRs. drmTMB #1663 merged after the 22:50 snapshot; its adjacent SD worktree remains dirty and owner-assigned. drmTMB #1659 has successful exact-head CI but a stale base and no live registered audit worktree. DRModels.jl #1242, #1239, and #1234 merged with recorded CI and documentation runs successful, while their known broken or skipped parity cells remain. #1240 remains blocked before its numerical test by missing fixture provenance. The 23:10 update records that no inspected candidate currently has both a verified live owner checkout and a collision-free path. #1248 has three failing shards on its exact head, with failure causes in existing temporal OU, coupled NB2, and all-singleton Gaussian checks; causal attribution to the PR is not established. #1657's exact-head CI is now fully green, but GitHub says it is non-mergeable and no owner checkout or review is recorded. #1249's Julia 1 shard 3 was still active at 23:09 UTC.

At 23:18 UTC, live main is `e5cd866` and #1657's submitted base remains `129c09c`; the compare is six ahead and 47 behind. Its API state is `dirty`/non-mergeable and there are no submitted reviews. R-CMD-check run 40375071745 failed only on the macOS release job at stale SHA `02095c67`; its log is no longer retrievable, and no later R-CMD-check appears in the latest 50 main workflow runs. #1249 shards 1, 2, and 4 passed while shard 3 remained in progress.

## 8. Consistency Audit

The rolling crosswalk labels the 22:50 census as historical and records the verified 23:00, 23:10, 23:12, and 23:18 updates. It distinguishes successful checks from mergeability and ownership, a merged PR from additional dirty work, and point-estimate agreement from recovery or interval coverage. The latest R-CMD-check failure is identified as stale rather than current-main evidence. CMR stays separate from the core DRM parity scope.

## 9. What Did Not Go Smoothly

GitHub state changed while the census was being reconciled: #1663 merged at 22:50:57 UTC, #1248's failed run became visible, and #1657's last active job completed successfully. Main then advanced, making #1657 47 commits behind and divergent; its prior green run no longer describes a current-base candidate. The separate capability-page worktree also had uncommitted edits with no verified current owner, so that page could not safely be consolidated in this slice. The owner refresh found no easy PR candidate safe to advance without an owner handoff.

## 10. Known Residuals

- The remaining open PRs still need exact-head check, base, owner, and overlap review before action.
- #1664 and several R candidates have green checks but GitHub reports them non-mergeable.
- #1214 and #1210 overlap across 55 files and need a canonical owner decision.
- The detached capability-status edits and their owner remain unresolved.
- No parameter-recovery or interval-calibration claim is added by this documentation refresh.
- #1657 requires its owner to refresh the 47-commit-stale base, rerun checks, and obtain review before any merge action. #1249 still has a running shard. #1248's failures need current-base or owner-led attribution before any fix is assigned.
- The latest R-CMD-check failure is on an older main SHA; its macOS log expired before the failure could be attributed. Current-main R-CMD-check status remains unverified.
- The full check-log validator remains unverified here because Julia is unavailable; older narrative-style entries also need separate owner-led cleanup before a whole-directory pass can be claimed.

## 11. Team Learning

Keep PR status counts tied to a timestamp, and report post-snapshot changes as an addendum. A merged pull request does not establish that later uncommitted work in its checkout is landed.

## 12. Cross-Product Coverage

Covers selected live PR and exact-head CI state in drmTMB and DRModels.jl, plus the observable owner and worktree state for #1663, the capability-page edits, and the current candidate disposition through 23:18 UTC. Does NOT cover all open PRs beyond the dated census, full package tests, source correctness review for every candidate, model parity, parameter recovery, interval calibration, merge, release, or deployment.
