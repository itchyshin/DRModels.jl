## 1. Goal

Reconcile the terminal CI and documentation jobs for DRModels.jl #1215 and update the parity crosswalk.

## 2. Implemented

- Verified the exact PR head and base.
- Inspected the completed CI and Documenter jobs, including their individual steps.
- Recorded what ran, what was skipped, and the unresolved documentation overlap.

## 3a. Decisions and Rejected Alternatives

- Counted the prose-only aggregate CI result as a successful docs-change check; Julia tests did not execute.
- Did not treat the docs build as preview or deployment evidence.
- Did not change the draft state, post a review, contact an owner, or merge.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-03-pr1215-terminal-check.md`
- `docs/dev-log/check-log.d/2026-11-03-pr1215-terminal-check.md`

## 5. Checks Run

- Refreshed exact-head metadata and workflow runs for DRModels.jl #1214, #1215, #1218, #1223, and #1249, plus drmTMB #1500 and #1667.
- Fetched job details for #1215 CI run `41539026157` and Documenter run `41539026225`.
- Confirmed CI aggregate success, with Julia setup/build/test steps skipped in all four shards; scaling sweep skipped.
- Confirmed Documenter build, rendered-page audit, and artifact upload succeeded; preview was cancelled and deploy skipped.
- Ran `git diff --check` after the documentation edit.
- No tests, fits, simulations, or package checks were run locally.

## 6. Tests of the Tests

Not applicable. This slice records hosted job evidence only.

## 7a. Issue Ledger

- #1215 remains a draft without formal review. Its ROADMAP change overlaps #1214 and #1210.
- #1214 remains in CI; #1249 remains queued/pending; #1667 remains in R-CMD-check. The other tracked run states were unchanged.
- No PR or issue metadata changed.

## 8. Consistency Audit

- Matched job details to #1215 head `1ee71530ebcd31bb62cce2fba6601ffced4f65ea` and base `973bbb675dc6a10b436acf86175e089e38ec467e`.
- Distinguished CI workflow success from skipped Julia tests, and docs build success from preview/deployment.
- Style: 2/10, medium confidence. Genre/coverage: operational after-task report, read in full. Evidence/repair: the report explicitly separates build, tests, preview, and deployment so readers can see which checks completed. Gates: science = not applicable; facts = pass against live GitHub PR metadata and job steps; references = not applicable because no external sources were cited. Provenance: self-review of this report, version 1, 2026-11-03; no held-out examples were used.

## 9. What Did Not Go Smoothly

The aggregate CI name is easy to overread: every Julia shard succeeded through a fast path that skipped the Julia test commands.

## 10. Known Residuals

- Julia tests did not run for #1215.
- The docs preview was cancelled, deployment was skipped, and PR ownership/overlap remain unresolved.
- This check refresh does not establish numerical parity or interval calibration.

## 11. Team Learning

Read workflow job steps when an aggregate check passes on a prose-only change. A green aggregate can represent a successful path that deliberately skipped runtime tests.

## 12. Cross-Product Coverage

Covers: current PR metadata and the exact #1215 hosted job steps.

Does NOT cover: runtime tests, package-wide validation, numerical parity, calibration, owner handoff, merge, release, publication, or deployment.
