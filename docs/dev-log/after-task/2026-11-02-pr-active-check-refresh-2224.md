## 1. Goal

Refresh the exact-head status of three PRs whose checks were active or unresolved.

## 2. Implemented

Updated the parity crosswalk with the 22:24 UTC state of DRModels.jl #1249 and drmTMB #1657 and #1659.

## 3a. Decisions and Rejected Alternatives

- Retained prior snapshots and added a timestamped refresh.
- Left owner-held branches and PR metadata unchanged.
- Did not run compute because GitHub already had the required exact-head jobs running.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-pr-active-check-refresh-2224.md`
- `docs/dev-log/check-log.d/2026-11-02-pr-active-check-refresh-2224.md`

## 5. Checks Run

- Fetched live PR metadata and check diagnostics for #1249, #1657, and #1659.
- Confirmed the status entries match each exact PR head.
- `git diff --check` and the after-task structure check passed.

## 6. Tests of the Tests

Not run. No package source or tests changed.

## 7a. Issue Ledger

#1249: shards 1 and 2 passed; shards 3 and 4 remain active. #1657: source-tree and OS-matrix passed, while Windows, macOS, and Ubuntu remain active; earlier same-head Ubuntu shards passed. #1659: Ubuntu/macOS passed and Windows remains cancelled on a stale base.

## 8. Consistency Audit

The report keeps passed, active, and cancelled states distinct and retains the exact bases and heads. It makes no merge recommendation.

## 9. What Did Not Go Smoothly

Lane preflight still reports a large active-lane census and a stale Cursor handover. The path-specific check found no competing edits to these crosswalk files; this slice changed only the requested status record.

## 10. Known Residuals

- #1249 and #1657 have running jobs.
- #1659 needs an owner-led base refresh and a successful Windows job-level check.
- Branch author metadata does not identify the live task owner.

## 11. Team Learning

Use exact-head check records for each status update, and retain same-head prior runs when the newest platform matrix is still active.

## 12. Cross-Product Coverage

Covers current PR metadata and CI state for one Julia and two R PRs. Does NOT cover source review, ownership handoff, model parity, package-wide acceptance, calibration, merge, release, or deployment.
