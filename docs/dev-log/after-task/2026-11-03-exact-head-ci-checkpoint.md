# After-task: exact-head CI checkpoint

## 1. Goal

Refresh the pending CI records for DRModels.jl #1214/#1249 and drmTMB #1667 after Totoro and DRAC became available.

## 2. Implemented

Added an exact-head checkpoint to the parity crosswalk. It records completed and pending shards without changing source, PR metadata, or CI.

## 3a. Decisions and Rejected Alternatives

- Kept the work to read-only CI reconciliation. Current jobs are already running on GitHub, so no Totoro or DRAC run was needed.
- Did not treat #1249 as validated while its four Julia shards remain active.
- Did not change PRs, contact owners, or merge anything.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md` (appended checkpoint)
- `docs/dev-log/check-log.d/2026-11-03-exact-head-ci-checkpoint.md` (created)
- `docs/dev-log/after-task/2026-11-03-exact-head-ci-checkpoint.md` (created)

## 5. Checks Run

- Read current GitHub check diagnostics for exact heads `5ba76b1` (#1214), `b959b18` (#1249), and `c133edb` (#1667).
- Confirmed #1214 shards 1 and 4 failed, shard 2 passed, and shard 3 remained active; docs preview passed.
- Confirmed all four #1249 Julia shards remained active; its Documenter build and preview passed.
- Confirmed #1667 source-tree and OS-matrix checks and Ubuntu shards 1, 2, and 4 passed; shard 3 remained active.
- Rechecked #1500, #1659, and #1215 status in GitHub's exact-head diagnostics.
- No package tests, simulations, fits, or compute-host jobs were run.
- `git diff --check`: passed.

## 6. Tests of the Tests

Not applicable. This change records upstream CI results and does not modify tests or implementation.

## 7a. Issue Ledger

- #1214 and #1249 remain open with unresolved exact-head checks.
- #1667 remains open with one Ubuntu shard active.
- No PR state or issue state was changed.

## 8. Consistency Audit

The crosswalk records exact revisions and the job-level states returned by GitHub. The recorded outcomes are not full green checks for #1214, #1249, or #1667. The red drmTMB README badge remains attributable to an older R-CMD-check run, not a current-main check.

## 9. What Did Not Go Smoothly

GitHub's check-diagnostics endpoint continued to report some jobs as active during this refresh, so final outcomes are not available yet.

## 10. Known Residuals

- #1214 has two failed shards and one still-running shard.
- #1249 has four still-running Julia shards.
- #1667 has one still-running Ubuntu shard.
- PR ownership, review disposition, and merge readiness remain unresolved.
- Totoro and DRAC availability has not been converted into a compute job; use them only if a focused reproduction is needed after CI is terminal.

## 11. Team Learning

For active CI investigations, retain the exact head SHA and per-shard status. A compute host does not replace a still-running hosted check when the question is the state of that exact CI run.

## 12. Cross-Product Coverage

Covers: status refresh for three active PR CI runs across drmTMB and DRModels.jl.

Does NOT cover: package correctness beyond those jobs, broad R–Julia parity, CMR, interval calibration, owner approval, or merge readiness.

## 13. Prose assessment

Perceived AI-like style: 2/10, medium confidence; this is a short operational evidence note. Genre and coverage: after-task report. Evidence and repair: statements are tied to exact-head GitHub checks. Gates: scientific assessment not applicable; factual checks pass against the recorded diagnostics; references are direct run records. Provenance: self-review of the saved draft on 2026-11-03.
