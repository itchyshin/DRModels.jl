## 1. Goal

Refresh exact-head CI and base status for the three PRs that still had active or unresolved checks.

## 2. Implemented

Recorded current diagnostics for DRModels.jl #1249 and drmTMB #1657 and #1659 in the dated parity crosswalk. Kept exact head and base SHAs, run IDs, and job conclusions attached to each finding.

## 3a. Decisions and Rejected Alternatives

- Kept the refreshed snapshot separate from older timestamps so the history remains auditable.
- Did not infer lane ownership from branch author or ask an owner to intervene.
- Did not run tests or remote compute; the task was to inspect existing CI evidence.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-pr-active-check-refresh.md`
- `docs/dev-log/check-log.d/2026-11-02-pr-active-check-refresh.md`

## 5. Checks Run

- Fetched live PR metadata and exact-head diagnostics for #1249, #1657, and #1659.
- Verified the returned check records carry each listed PR head SHA.
- `git diff --check` and the repository after-task structure check passed.

## 6. Tests of the Tests

Not run. No source or test code changed; this slice records CI state only.

## 7a. Issue Ledger

#1249 has three Julia shards still running. #1657 has its current OS-matrix jobs still running, while the four Ubuntu shards from an earlier same-head run passed. #1659 still needs a current-main rebase and a successful Windows job-level check.

## 8. Consistency Audit

The new section distinguishes completed, in-progress, and cancelled jobs and preserves exact bases. A successful inner Windows step is not promoted to a successful parent job. No merge recommendation is made.

## 9. What Did Not Go Smoothly

The repository lane census still reports many active lanes and an old Cursor handover. The work remained limited to the explicitly requested PR-status crosswalk; no concurrent source or PR branch was touched.

## 10. Known Residuals

- Current CI for #1249 and #1657 is incomplete.
- #1659 remains on a stale base with Windows cancelled.
- Formal branch records do not identify the live human or agent owner.

## 11. Team Learning

Record CI at the PR head, and keep a job's conclusion separate from conclusions of its inner steps. Preserve earlier same-head runs when a later matrix is still running.

## 12. Cross-Product Coverage

Covers metadata and CI state for three open PRs across drmTMB and DRModels.jl. Does NOT cover source review, owner confirmation, code changes, package tests, model parity, calibration, merge, release, or deployment.
