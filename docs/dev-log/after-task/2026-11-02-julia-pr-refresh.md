## 1. Goal

Refresh the exact-head state of four additional DRModels.jl PRs and identify reviewable next steps without taking owner-held branches.

## 2. Implemented

- Refreshed PR metadata and exact-head check diagnostics for #1228, #1237, #1240, and #1247.
- Confirmed DRModels.jl main remains `973bbb675dc6a10b436acf86175e089e38ec467e`.
- Added a comparison table with bases, current checks, and owner-safe next steps to the parity crosswalk.
- Recorded the prior 22:11:46 UTC state of #1249 without polling its active shards too soon.

## 3a. Decisions and Rejected Alternatives

- Did not edit branches, update PR metadata, or contact owners.
- Kept stale-base, failed, and cancelled results distinct from successful checks.
- Did not run local tests while exact-head CI provided current evidence.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-julia-pr-refresh.md`
- `docs/dev-log/check-log.d/2026-11-02-julia-pr-refresh.md`

The existing modified Python bytecode file remains untouched.

## 5. Checks Run

- GitHub PR metadata and exact-head check diagnostics for four DRModels.jl PRs.
- Direct read-only lookup of the live `main` ref.
- `git diff --check`, after-task structure check, and `slop_check.py` for this report.
- No package tests, fits, simulations, or benchmarks were run.

## 6. Tests of the Tests

The external CI states apply to the named PR heads. This slice changes evidence documentation only.

## 7a. Issue Ledger

- #1228, #1237, #1240, and #1247 remain open.
- No issues or PR metadata were changed.

## 8. Consistency Audit

- Matched each check result to its exact PR head and recorded base SHA.
- Marked bases stale against current main.
- Preserved the distinction between reported local test results, GitHub checks, and interval-validation evidence.
- Left other lanes' PR branches and source files untouched.

## 9. What Did Not Go Smoothly

Several PRs have stale bases and incomplete or failed exact-head checks. No owner handoff was present for taking their branches.

## 10. Known Residuals

- #1228 needs a narrower BetaBinomial statement and fresh docs preview.
- #1237 needs its Julia 1.10 failures diagnosed after rebase.
- #1240 needs shard 2 diagnosed after rebase.
- #1247 needs a rerun after #1249 and separate attribution of the all-singleton Hessian case.
- Numerical parity and uncertainty validation remain broader unfinished goals.

## 11. Team Learning

A green earlier run does not cancel a later failed exact-head run. Keep the most recent run and its precise failing shards visible, and distinguish a branch author's identity from live lane ownership.

## 12. Cross-Product Coverage

Covers: live PR metadata and check states for four Julia PRs, with #1249's last known active state as context.

Does NOT cover: code correctness beyond the listed test suites, parity campaigns, uncertainty calibration, merge, release, or publication.

Editorial check: `slop_check.py` reported no findings for this exact report, 2026-11-02. Science: limited to the stated validation scope. Facts: limited to GitHub PR metadata and diagnostics. References: no external sources used.
