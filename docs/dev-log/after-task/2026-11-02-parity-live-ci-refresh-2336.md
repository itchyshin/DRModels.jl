# After-task: live parity CI refresh

## 1. Goal

Refresh selected drmTMB and DRModels.jl pull request, CI, and owner evidence in the parity crosswalk. Keep exact-head checks separate from current-main validation and leave owner-held branches unchanged.

## 2. Implemented

Added a timestamped 23:36 UTC update to the crosswalk. It records that DRModels.jl #1249 still has Julia shard 3 running, drmTMB #1500 remains in progress, drmTMB #1563 passed exact-head checks on an old base, and drmTMB #1659 is green on an old base. It confirms the latest drmTMB main-branch R-CMD-check is still the October 26 failure on an earlier source SHA.

## 3a. Decisions and Rejected Alternatives

- Did not infer lane ownership from a branch prefix, PR author, or green CI.
- Did not rebase, edit, comment on, merge, or close any owner-held PR.
- Stopped the exact-source Mac R CMD check during its build phase after finding another long R test active. It did not reach package tests, and no check result is claimed.
- Preserved the separate crosswalk branch's after-task changes. This refresh uses a new uniquely named close-out file rather than overwriting that report.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-parity-live-ci-refresh-2336.md`
- `docs/dev-log/check-log.d/2026-11-02-parity-live-ci-refresh-2336.md`

The unrelated modified `tools/__pycache__/runtests_files.cpython-313.pyc` was preserved.

## 5. Checks Run

- Refreshed GitHub PR, branch, workflow, and job metadata through `gh`.
- Checked the exact current main SHAs and the local DRModels.jl worktree list.
- Started an exact-main archive/build in a temporary directory, then interrupted it before `R CMD check` began to avoid competing with the active R lane.
- `git diff --check` passed for the edited crosswalk.
- The prose checker reported zero findings for the crosswalk.
- The check-log contains one row and matches the documented row format. Julia is unavailable here, so the combined check-log renderer was not run.
- No package tests, simulation, or fits were run. The interrupted archive/build is not validation evidence.

## 6. Tests of the Tests

Not applicable. This was a documentation and evidence refresh, not a test or implementation change.

## 7a. Issue Ledger

- DRModels.jl #1249: shard 3 remains active; live logs unavailable until it completes; no formal review or exact-head local checkout was found.
- drmTMB #1500: four Ubuntu shards remain active at the observed time; no formal review was recorded.
- drmTMB #1563: exact-head checks passed, but the base is stale and GitHub reports the PR dirty against current main.
- drmTMB #1659: exact-head checks passed, but the base is stale and there is no formal review.
- Current-main R-CMD-check remains unverified. The latest main workflow run remains an October 26 failure on stale source.

## 8. Consistency Audit

The crosswalk names current main as drmTMB `e5cd86667c97d1b34f0ead22017a675377f46930` and DRModels.jl `973bbb675dc6a10b436acf86175e089e38ec467e`. Each PR status is tied to its observed head SHA and run ID. No exact-head result is presented as current-main validation.

## 9. What Did Not Go Smoothly

GitHub withheld the log for the still-running #1249 shard. Another long R test was active on the Mac, so the exact-source archive/build was interrupted before the package check began. The lane owner for #1249 and current owners for #1500 and #1563 were not independently confirmed.

## 10. Known Residuals

- #1249 shard 3 and #1500's four shards need terminal results and later owner review.
- #1563 and #1659 need current-base reconciliation by their owners before their green checks can support integration.
- A current-main drmTMB R-CMD-check is still required to diagnose the public failing badge.
- This refresh does not audit all open PRs, establish package parity, or validate interval coverage.

## 11. Team Learning

Record the commit and job status with every CI claim. Treat the PR branch prefix as branch provenance, not proof of who currently owns the lane.

## 12. Cross-Product Coverage

Covers selected live PR and CI metadata for both repositories through 23:36 UTC. Does not cover every open PR, code correctness, package-wide test status, R-Julia numerical parity, CMR interval calibration, or merge readiness.
