# After-task: attribute the drmTMB R-CMD-check badge

## 1. Goal

Determine what current evidence explains the red R-CMD-check badge on drmTMB's README.

## 2. Implemented

Recorded the current main head, latest main-branch R-CMD-check run, its job-level results, and the distinction from the separate current-main receipt-staleness failure.

## 3a. Decisions and Rejected Alternatives

- Treated run `40375071745` as historical because it tested head `02095c6`, while current main is `e5cd866`.
- Did not claim that current main fails R-CMD-check. No R-CMD-check run is listed for the current head.
- Did not start a current-main check or infer the old macOS failure cause. The old job log endpoint returns HTTP 410.
- Did not change the R checkout, workflow, badge, or any remote state.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md` (modified)
- `docs/dev-log/after-task/2026-11-03-r-cmd-badge-status.md` (created)
- `docs/dev-log/check-log.d/2026-11-03-r-cmd-badge-status.md` (created)

## 5. Checks Run

- Read current `drmTMB` main branch metadata and workflow runs through the GitHub connector.
- Compared the README badge target at cached `origin/main` with the live main head.
- Inspected the job list for run `40375071745`: macOS failed; source-tree, Ubuntu, OS-matrix, and Windows jobs passed.
- Confirmed `receipt-staleness` run `41540682271` failed on current head `e5cd866` as a separate workflow.
- Retrieved the old macOS job log; GitHub returned HTTP 410. The public run page identifies the failed platform and status, while the logs require sign-in.
- `git diff --check`: passed.
- `Rscript ~/shinichi-brain/tools/check-after-task.R docs/dev-log/after-task/2026-11-03-r-cmd-badge-status.md`: passed.
- `python3 ~/shinichi-brain/tools/slop_check.py <absolute after-task path>`: zero findings.

## 6. Tests of the Tests

No package tests were run. The Actions run list establishes that no R-CMD-check run is listed for current main; it does not establish that current main would pass a new check.

## 7a. Issue Ledger

- No issue or PR was opened or closed.
- No workflow was dispatched and no owner was contacted.

## 8. Consistency Audit

The README badge targets the R-CMD-check workflow. The latest listed main run is `40375071745` at `02095c6`; the current main head is `e5cd866`. The only push-triggered check run reported on `e5cd866` is the failed `receipt-staleness` workflow, which does not determine the R-CMD-check badge.

## 9. What Did Not Go Smoothly

The historical job logs expired before this review, preventing attribution of the macOS check failure.

## 10. Known Residuals

- Current-main R-CMD-check has not been run or verified.
- The cause of the October 2026 macOS job failure remains unknown.
- Current main also has a separate failed receipt-staleness check; it is outside this badge attribution.

## 11. Team Learning

A red workflow badge can remain attached to a historical run after main advances. Record the run head and platform jobs before describing the current source as failing.

## 12. Cross-Product Coverage

Covers: status attribution for the drmTMB README's R-CMD-check badge.

Does NOT cover: current-main package correctness, the receipt-staleness failure, DRModels.jl parity, CMR, interval calibration, or a release decision.

## 13. Prose assessment

Perceived AI-like style: 2/10, medium confidence; the report is a short status attribution with exact commit and run identifiers. Genre and coverage: self-review of the full after-task report. Evidence and repair: no generic background is needed for this technical note; no further repair is needed. Gates: science not applicable; facts pass against the branch, run, and job data recorded above; references not applicable because there are no literature claims. Provenance: self-review of this saved draft on 2026-11-03; prior crosswalk reports were seen during this task.
