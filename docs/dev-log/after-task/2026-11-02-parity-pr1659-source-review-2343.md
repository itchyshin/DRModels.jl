# After-task: drmTMB #1659 source review

## 1. Goal

Review the green parity-fixture PR #1659 against its exact GitHub diff and current PR metadata. Identify what its evidence supports and any provenance limit without modifying the owner-held branch.

## 2. Implemented

Added a focused source-review note to the parity crosswalk. It records the exact PR head/base, successful exact-head checks, the historical DRModels.jl fixture pin, the conditional live bridge layer, and a provenance issue in the reference generator's dirty-tree check.

## 3a. Decisions and Rejected Alternatives

- Did not infer current lane ownership from the branch author.
- Did not post a review or comment, edit the PR, rebase it, or merge it.
- Did not rerun the full R package suite. This was a static diff and metadata review.
- Kept the provenance finding separate from the numerical parity claim; it does not by itself invalidate the stored estimates.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-parity-pr1659-source-review-2343.md`
- `docs/dev-log/check-log.d/2026-11-02-parity-pr1659-source-review-2343.md`

The unrelated modified bytecode file remains untouched.

## 5. Checks Run

- Read the GitHub PR file list, exact head/base, review list, and six exact-head check results.
- Inspected the added test file and generator diff from the exact PR head.
- `git diff --check` and prose checks passed.
- The check-log row follows the documented one-row format. Julia is unavailable here, so the combined renderer was not run.
- No Julia or R package tests were run.

## 6. Tests of the Tests

Not applicable. No test or implementation code was changed.

## 7a. Issue Ledger

- PR #1659 has six successful exact-head checks, but the base predates current drmTMB main and there are no formal reviews.
- The provenance generator's `--untracked-files=no` status query can report a clean state despite untracked files in the Julia checkout.
- The owner should clarify or strengthen the provenance check before interpreting `drmjl_dirty: false` as a clean full checkout.

## 8. Consistency Audit

The crosswalk distinguishes the historical fixture commit `64ccb2d...` from current DRModels.jl main `973bbb...`. It also distinguishes the stored TMB comparison from the conditional live R-to-Julia layer. The provenance caveat is limited to clean-tree reporting and is not described as a numerical failure.

## 9. What Did Not Go Smoothly

The PR's exact-head CI does not establish validation on current drmTMB main. Its live bridge test is conditional, and the CI record reviewed here did not prove that layer ran. No exact-head local checkout or current lane owner was confirmed.

## 10. Known Residuals

- The provenance check needs owner disposition.
- The PR needs a current-base rerun and formal review before integration claims.
- This static review does not validate all model routes, recovery, interval calibration, or complete R-Julia parity.

## 11. Team Learning

Record whether a provenance dirty flag includes untracked files; a tracked-only status query cannot support a full clean-tree claim.

## 12. Cross-Product Coverage

Covers the source and evidence claims of drmTMB #1659. Does NOT cover all open PRs, current-main package checks, broad model parity, or interval calibration.
