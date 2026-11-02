## 1. Goal

Refresh owner-attributed status for additional open parity-related PRs across the R and Julia repositories.

## 2. Implemented

Added exact-head CI and ownership caveats for drmTMB #1663, #1647, and #1435, plus DRModels.jl #1215, #1214, and #1210, to the rolling crosswalk. No PR or source branch was changed.

## 3a. Decisions and Rejected Alternatives

- Kept the #1214/#1210 overlap as an owner decision instead of choosing a patch from file overlap alone.
- Kept #1435 closed to action until an owner resolves its stale base and current ownership.
- Left the dirty exact-head #1663 worktree untouched.
- Did not use Totoro or DRAC because this slice only read GitHub metadata and local checkout records.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-pr-active-check-refresh-2245.md`
- `docs/dev-log/check-log.d/2026-11-02-pr-active-check-refresh-2245.md`

## 5. Checks Run

- Refreshed exact PR heads, bases, mergeability, and check runs for the listed candidates.
- Used read-only R and Julia PR census agents to compare open PRs with current local worktrees; no agent edited or contacted anyone.
- `git diff --check`, after-task structure validation, and the prose slop check passed.

## 6. Tests of the Tests

Not run. No source or test code changed.

## 7a. Issue Ledger

#1663 has three of four Ubuntu shards passed and one active, with an exact-head dirty owner worktree. #1647 has Ubuntu and macOS passed and Windows active. #1435 is green on a stale base with current ownership unconfirmed. #1215 is a one-file draft with checks queued and no matching current-head checkout. #1214 and #1210 overlap across 55 files and need a canonical patch decision.

## 8. Consistency Audit

The crosswalk distinguishes current-head CI, older same-head CI, stale bases, queued checks, and local checkout evidence. It does not infer task ownership from branch names or PR authors.

## 9. What Did Not Go Smoothly

The repositories have many live lanes. Several apparent cleanup candidates either have checks still running, stale bases, or overlapping edits. No safe branch mutation was supported by this read-only census.

## 10. Known Residuals

- #1663 shard 4 and #1647 Windows remain active.
- #1435 requires owner confirmation and a current-main check run.
- #1215 requires its queued checks and a current owner checkout.
- #1214 and #1210 need an owner decision on the canonical patch.

## 11. Team Learning

Use current head and base SHAs with check runs, then check for a matching dirty worktree before treating a green PR as available cleanup work.

## 12. Cross-Product Coverage

Covers six additional parity-related PRs and their latest available ownership and CI evidence. Does NOT cover full package tests, model parity, recovery, interval calibration, PR source review, merge, release, or deployment.
