# After-task: attribute the current #1214 CI failures

## 1. Goal

Inspect the failed exact-head jobs for DRModels.jl #1214, compare them with active fixes, and update the parity crosswalk without changing another lane's source or PR.

## 2. Implemented

Added a timestamped crosswalk entry with the two failing assertions, relevant #1249 source and test changes, exact heads, and the remaining live-run state. Added one check-log row and this report.

## 3a. Decisions and Rejected Alternatives

- Kept #1214 and #1249 open and unchanged. The owners have not supplied a handoff, and #1249 has not completed exact-head checks.
- Did not call the #1214 failures documentation-caused. The failures are in Gaussian location-scale behavior and an optimizer tolerance test, and #1249 independently changes those areas.
- Assumed the user authorized a crosswalk-only update through the existing parity reconciliation task. If that interpretation is wrong, revert only this additive crosswalk entry and its two report files.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md` (appended evidence)
- `docs/dev-log/check-log.d/2026-11-03-pr1214-ci-failure-attribution.md` (created)
- `docs/dev-log/after-task/2026-11-03-pr1214-ci-failure-attribution.md` (created)

## 5. Checks Run

- `gh pr view` and `gh run view` checked current PR heads, base, and job states.
- GitHub job-log retrieval for #1214 jobs `125614277806` and `125614277774` identified failures in `test/test_ls_inv2x2_chol.jl:123` and `test/test_gaussian_ranef.jl:59-62`.
- Compared the failure sites with the current #1249 patch. It changes `src/gaussian_ranef.jl` and the tolerance assertion in `test/test_ls_inv2x2_chol.jl`.
- Refreshed the same runs at 00:34 UTC: #1214 shard 2 passed and shard 3 was still running; all four #1249 Julia shards were still running.
- `git diff --check`: passed.
- No tests, simulations, fits, or compute-host jobs were run.

## 6. Tests of the Tests

Not applicable. This slice records and attributes existing CI results; it changes no test or implementation behavior. The upstream test outcomes remain the evidence for their respective assertions.

## 7a. Issue Ledger

- #1214 remains open. Two of four Julia shards were still active at the recorded refresh; two failed as described.
- #1249 remains open. Its test and source edits target the observed failures, but its Julia shards were queued or running at the recorded refresh.
- Current ownership is unconfirmed. No comments, reviews, or PR metadata were changed.

## 8. Consistency Audit

Checked the #1214 and #1249 exact heads and common base against the remote PR records. The #1249 diff contains changes at both observed failure sites. Its current run is not yet terminal, so this is target correspondence, not proof of repair. The #1214 docs build and preview passed, but package CI is not yet complete.

## 9. What Did Not Go Smoothly

`gh run view --log-failed` did not provide logs while the overall run was active. Fetching the two already-finished job logs by their job IDs worked, allowing failure attribution without waiting for the remaining shards.

## 10. Known Residuals

- #1214 shard 2 passed at 00:32 UTC; shard 3 was still running at the 00:34 UTC refresh.
- #1249's docs preview and Julia shards were queued or running at that observation. Its exact-head suite needs a fresh check after completion.
- The broader R–Julia parity goal still needs a route-by-route acceptance table and cannot be estimated from these documentation PR checks.
- No owner disposition or merge recommendation is established.

## 11. Team Learning

Memory receipt: `route.py` found no LOAD-FIRST manifest for this worktree. The project instructions and efficiency protocol shaped the narrow, exact-head evidence refresh. The general memory note on checking Totoro load before a campaign was not used because this work required no compute. Golden Set: not in scope. Durable lesson: an active workflow can still expose logs for completed jobs through the job-log endpoint; record the job IDs and inspect completed failures while unrelated shards continue.

## 12. Cross-Product Coverage

Covers: the two reported Julia 1.13.3 Linux failures in the exact #1214 run, compared with their corresponding candidate changes in #1249.

Does NOT cover: the remaining #1214 shards, #1249's final checks, runtime behavior outside CI, R package parity, statistical calibration, ownership handoff, or merge readiness.
