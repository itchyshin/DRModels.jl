## 1. Goal

Refresh exact-head checks for active parity PRs and make the canonical PR status record easy to find from the capability ledger.

## 2. Implemented

Added the 22:40 UTC exact-head status for DRModels.jl #1249 and drmTMB #1657, #1659, and #1664 to the rolling crosswalk. Added a pointer from the capability ledger to the rolling crosswalk and labelled that ledger's PR rows as dated snapshots.

## 3a. Decisions and Rejected Alternatives

- Kept earlier snapshots as history instead of deleting them.
- Left all PR branches, metadata, and owner-held source unchanged.
- Did not run compute because GitHub had the required checks running and no simulation or benchmark was part of this slice.

## 4. Files Touched

- `docs/design/capability-status.md`
- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-pr-active-check-refresh-2240.md`
- `docs/dev-log/check-log.d/2026-11-02-pr-active-check-refresh-2240.md`

## 5. Checks Run

- Refreshed PR metadata and exact-head check runs from GitHub for #1249, #1657, #1659, and #1664.
- Checked formal review submissions and review threads for those PRs; none were recorded.
- `git diff --check`, the repository after-task structure check, and the writing slop check passed.

## 6. Tests of the Tests

Not run. No package source or test code changed.

## 7a. Issue Ledger

#1249: Julia shards 1, 2, and 4 passed; shard 3 remains active on current main. #1657: source-tree and OS-matrix checks passed; current platform jobs remain active, and four earlier same-head Ubuntu shards passed. #1659: Ubuntu and macOS passed, Windows is cancelled, and the base is stale. #1664: all four Ubuntu shards, OS-matrix, and source-tree checks passed on current main; no formal review is recorded.

## 8. Consistency Audit

The report distinguishes active, passed, cancelled, and skipped checks. It does not equate job-internal success with a successful parent job, and it does not treat green CI as a merge decision.

## 9. What Did Not Go Smoothly

The file-level preflight reported four refs with capability-page edits and a large live-lane census. The user had authorized taking this page after diff review. I reviewed the overlapping ref summaries and retained the existing dated history, making only a pointer to the rolling status record.

## 10. Known Residuals

- #1249 shard 3 and the current #1657 platform jobs are still running.
- #1659 needs an owner-led base refresh and a successful Windows job-level check.
- #1664 is ready for maintainer review, but no GitHub review has been submitted.
- No PR was created for this crosswalk branch, and no merge, release, or deployment occurred.

## 11. Team Learning

Keep capability claims in the design ledger and volatile PR state in a dated evidence record, with a direct link between them.

## 12. Cross-Product Coverage

Covers current GitHub status for four parity-related PRs and navigation to that evidence. Does not cover full-package acceptance, model parity, recovery, interval calibration, CMR readiness, merging, release, or deployment.
