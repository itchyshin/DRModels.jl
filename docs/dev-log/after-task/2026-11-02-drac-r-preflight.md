## 1. Goal

Refresh the current parity PR status and determine whether restored DRAC access supports a bounded current-main drmTMB R CMD check.

## 2. Implemented

- Refreshed exact-head CI and review state for DRModels.jl #1249 and drmTMB #1500.
- Verified the Rorqual SSH ControlMaster, current Slurm queue, R version, library paths, and installed dependency availability.
- Recorded the result in `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`.

## 3a. Decisions and Rejected Alternatives

- Kept the existing R library read-only and did not modify the installed drmTMB tree.
- Did not start R CMD check because required Imports and LinkingTo packages are missing.
- Did not install dependencies or submit a Slurm job. Setup duration and package-build feasibility have not been measured.
- Did not touch PR branches, request owner changes, or merge.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-drac-r-preflight.md`
- `docs/dev-log/check-log.d/2026-11-02-drac-r-preflight.md`

The pre-existing modified Python bytecode file was left untouched.

## 5. Checks Run

- Read the repository coordination board and the latest parity crosswalk receipt.
- Exact-head GitHub metadata and workflow jobs were read for #1249 and #1500. #1249 shards 1, 2, and 4 passed; shard 3 remained active. #1500 source-tree and OS-matrix jobs passed; all four Ubuntu shards remained active.
- Rorqual R 4.6.1 preflight found 30 installed packages in the selected user and system libraries. Three of eight Imports, both LinkingTo packages, and 27 of 29 Suggests packages were absent. `pak` was absent.
- `squeue` showed no jobs for the account user. The existing library was not written.
- `git diff --check` passed after the report and check-log entry were written.
- `check-after-task.R` passed. `slop_check.py` reported zero findings on the exact report. Julia was unavailable, so `tools/build_check_log.jl --check` was not run.
- No package tests, model fits, simulations, or benchmarks were run.

## 6. Tests of the Tests

Not applicable. This was a read-only runtime and dependency preflight plus an evidence update. No test implementation changed.

## 7a. Issue Ledger

- #1249 and #1500 remain open; neither has a formal review. Their branches remain unchanged.
- No PR or issue metadata was changed.

## 8. Consistency Audit

- The crosswalk records exact PR heads and bases returned by GitHub.
- The R dependency counts were separated by DESCRIPTION field, so missing Imports and LinkingTo are distinguished from missing Suggests.
- Rorqual availability is distinguished from a ready R check environment.
- No inference was made that an active GitHub job had failed because its log endpoint returned 404.

## 9. What Did Not Go Smoothly

The first dependency-reporting expression assumed a different return shape from R's internal dependency parser and stopped before printing counts. A read-only structure inspection corrected the query; the final per-field counts were then returned successfully.

## 10. Known Residuals

- A current-main Linux R CMD check remains unrun. An isolated package library must first be prepared and its build time estimated.
- #1249 shard 3 and #1500's four Ubuntu shards were still active at the last exact-head refresh.
- The broader R-Julia parity backlog and CMR package separation remain separate open work.

## 11. Team Learning

When a compute host returns, verify its software environment as well as its connection. A reachable host and empty queue do not establish that the required package toolchain is ready.

## 12. Cross-Product Coverage

Covers: current exact-head status for two PRs and Rorqual readiness for a current-main R CMD check.

Does NOT cover: R CMD check results, R-Julia numerical parity, interval calibration, PR ownership, merge, release, publication, or deployment.

Editorial check: `slop_check.py` reported zero findings for this exact report on 2026-11-02. Science: not applicable. Facts: checked against current GitHub metadata and read-only Rorqual output. References: GitHub run links are recorded in the crosswalk.
