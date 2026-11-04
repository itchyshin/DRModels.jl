## 1. Goal

Refresh the full open-PR inventory for drmTMB and DRModels.jl from current GitHub, exact-head CI, current main refs, and registered worktree evidence. Preserve owner and scope boundaries, including the separate CMR programme.

## 2. Implemented

Added a dated evidence census with exact head/base SHAs for 33 open drmTMB PRs and 22 open DRModels.jl PRs at the captured search time. Added read-only inventories of all 83 drmTMB and 90 DRModels.jl worktree registrations, including dirty/diverged clusters, stale branch-only PR matches, and owner uncertainty. Twelve Julia worktree paths were unavailable, so their status remains unknown. Confirmed that the SD-regression merges are complete and that G5 calibration remains deferred.

## 3a. Decisions and Rejected Alternatives

- Kept the full PR list in a dedicated dated evidence file, separate from the shorter crosswalk README changed by the open #1252 lane.
- Did not infer a current owner from PR author, branch prefix, mergeability, or a stale local checkout.
- Did not rebase or edit PR branches, rerun CI, comment, merge, release, publish, or deploy.
- Kept CMR #1612 and its deferred interval calibration separate from the DRM core.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-04-open-pr-census/README.md`
- `docs/dev-log/check-log.d/2026-11-04-open-pr-census-refresh.md`
- `docs/dev-log/after-task/2026-11-04-open-pr-census-refresh.md`

## 5. Checks Run

- Queried GitHub for all open PRs in both repositories; counts and PR numbers were independently listed from live search.
- Read full PR metadata for all 22 DRModels.jl PRs. The R exact head/base and check inventory was returned by the read-only census audit for all 33 R PRs.
- Rechecked both current main branch SHAs at 15:43 UTC.
- Rechecked DRModels.jl #1252 exact-head diagnostics at 15:43 UTC, including the still-running Julia shard 4.
- Inspected the original crosswalk worktree and the clean #1252 worktree; left both untouched.
- Scanned all registered worktrees in both repositories. No source, ref, worktree, lease, or PR state was changed.
- Compared the dirty crosswalk README and local after-task report with #1252's exact tree. The PR contains every local line plus additional text; the old worktree remains intact, including an unrelated pycache edit.
- Refreshed #1252 and #1253 status at 16:11 UTC. #1252 shard 3 had failed and shard 4 was still running; #1253 checks were queued on its recorded head.
- Package tests, fits, simulations, site builds, and R-CMD-check were not run by this documentation-only slice.

## 6. Tests of the Tests

Not run. No package source or tests changed. Row-count and metadata consistency checks are recorded below.

## 7a. Issue Ledger

- The 2026-11-02 census is superseded in count by this snapshot: 33 R and 22 Julia PRs are currently open.
- No reviewed candidate is both current-base, exact-head checked, formally reviewed, and owner-free.
- The #1252 docs preview passed, but shard 3 failed and shard 4 remained in progress at the last check. It is not ready for an owner-safe merge.
- drmTMB current main has a successful receipt-staleness workflow but no R-CMD-check run on its current SHA.
- D-342 and the disclosed tolerance and numerical/export limits remain open. Calibration G5 remains deferred.

## 8. Consistency Audit

The evidence distinguishes current main from PR base, exact-head checks from historical checks, formal review from branch author, and a registered worktree from a confirmed owner. The R snapshot is timestamped 15:36 UTC; the Julia snapshot is timestamped 15:34 UTC and #1252 was separately refreshed at 15:43 UTC. No completion, parity, or calibration claim is inferred from mergeability or a documentation build.

## 9. What Did Not Go Smoothly

The lane census reports 32 live R lanes and 23 live Julia lanes. Many PRs have stale or stacked bases, missing exact-head checks, conflicts, failed checks, or unconfirmed owners. The worktree inventory found no current owner release for dirty or diverged work. No source or PR takeover was made.

## 10. Known Residuals

- At 16:11 UTC, #1252 shard 4 remained running while shard 3 had failed on the current PR head. Refresh the same run after completion.
- This census records PR metadata and worktree state. Review source diffs and refresh changed heads and checks before taking action.
- The 12 missing Julia worktree registrations remain unknown. Preserve the CMR, SD, covariance, and parity bundles until their owners release them.
- Julia's live open-PR count is now 23 because #1253 opened after the 22-PR metadata snapshot.
- No current-main R-CMD-check receipt exists for drmTMB.
- The broader R–Julia parity goal remains open. CMR calibration G5, D-342, two disclosed tolerance adjustments, and known numerical/export limits remain unresolved.
- The original crosswalk branch and its dirty files remain preserved. The existing #1252 crosswalk lane owns its README.

## 11. Team Learning

A complete PR inventory is useful only when exact heads, bases, current checks, worktree evidence, and confirmed ownership are reported separately. `mergeable=true` is not a merge recommendation.

### Post-draft assessment

The report uses concrete repository states and limits each claim to its timestamped evidence. Genre and coverage: factual evidence report and after-task record; full text reviewed. Evidence and repair: “no owner-free” and “current check” are bounded by timestamp and reviewed PR scope; wording was tightened to state package and source-review limits directly. Gates: science = not applicable, no scientific result claimed; facts = pass within the listed GitHub and worktree records, with timestamp limits stated; references = pass, links point to repository PR and run pages. Provenance: review of the 2026-11-04 draft, including earlier census wording.

## 12. Cross-Product Coverage

Covers timestamped GitHub PR metadata, registered-worktree counts and ownership limits, main refs, selected lane evidence, merged SD receipt closeout, and the separate CMR boundary. It does not cover all PR source diffs, a full package suite, current-main R-CMD-check, R–Julia numerical parity, recovery, interval calibration, or authorization to merge or publish.
