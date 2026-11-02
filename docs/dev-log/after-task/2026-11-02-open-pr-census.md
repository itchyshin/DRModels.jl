## 1. Goal

Build a current, owner-aware inventory of open drmTMB and DRModels.jl PRs to support safe cleanup and future parity work.

## 2. Implemented

Recorded all 35 open drmTMB PRs and 24 open DRModels.jl PRs, including exact head/base SHAs, check states, formal review state, registered worktree evidence, and major scope conflicts. Updated the parity crosswalk with the census summary and the completion of drmTMB #1663 checks.

## 3a. Decisions and Rejected Alternatives

- Kept each repository's full inventory in a dedicated evidence file rather than inflating the shorter parity crosswalk.
- Did not treat author, branch name, mergeability, or a stale registered checkout as proof of active owner or permission to edit.
- Kept CMR #1612 distinct from DRM core and its deferred interval-calibration programme.
- Did not run package tests or compute because this slice classifies GitHub metadata and local worktree records only.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-open-pr-census/README.md`
- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-open-pr-census.md`
- `docs/dev-log/check-log.d/2026-11-02-open-pr-census.md`

## 5. Checks Run

- Read-only PR metadata, exact-head check runs, review submissions, and review threads for all open PRs in both repositories.
- Read-only registered-worktree and conflict-scope comparisons from the two repository censuses.
- The two census agents made no file, PR, branch, or review changes.
- `git diff --check`, the after-task structure check, and the prose slop check passed.
- A row-count check confirmed 35 R PR rows and 24 Julia PR rows.

## 6. Tests of the Tests

Not run. No package source or test code changed.

## 7a. Issue Ledger

The census found 59 open PRs and no submitted formal review across either repository. #1664 is a small green current-main PR ready for maintainer review. #1663 is green but has a heavily dirty exact-head worktree. #1435 is small but stale-base. #1215 is the smallest Julia documentation candidate, but checks are queued and its checkout is stale. #1214/#1210 and several R PR groups have overlapping files requiring owner coordination.

## 8. Consistency Audit

The full census separates exact-head checks from old runs, current bases from stale or stacked bases, registered checkouts from current heads, and check state from owner identity. It records active and cancelled jobs without counting them as successful.

## 9. What Did Not Go Smoothly

Most open PRs have no formal review, and many have stale or stacked bases. Several local worktrees are dirty or behind their PR heads. No safe branch mutation or merge was supported by this census.

## 10. Known Residuals

- Ownership remains unconfirmed for most PRs.
- Running and queued jobs need a later exact-head refresh.
- No PR is both current-main, fully green, conflict-free, formally reviewed, and ready for owner-safe merge.
- The full DRM parity goal remains open; this is an organization snapshot, not parity validation.

## 11. Team Learning

A complete PR census is useful only when checks, bases, worktrees, review state, and file conflicts are kept in separate evidence fields.

## 12. Cross-Product Coverage

Covers the current open PR and registered-worktree inventory for both repositories. Does NOT cover source-level review of every PR, full package acceptance, current R–Julia numerical parity, parameter recovery, interval calibration, merge, release, or deployment.
