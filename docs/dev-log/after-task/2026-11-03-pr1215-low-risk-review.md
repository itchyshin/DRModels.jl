## 1. Goal

Update the parity crosswalk with exact-head follow-up for three active PRs and a source review of the smallest current-main documentation candidate.

## 2. Implemented

- Refreshed exact-head workflow jobs for #1249, #1500, #1667, and #1215.
- Reviewed the #1500 diff read-only and checked the #1215 link targets against its exact base.
- Compared #1215's changed path with the overlapping documentation PRs #1214 and #1210, then recorded owner and checkout limits without changing PR branches or metadata.

## 3a. Decisions and Rejected Alternatives

- Did not run cluster tests because exact-head CI was active or had just completed the relevant checks.
- Did not mark drafts ready, post reviews, contact owners, merge, or change remote metadata.
- Assumption: prose-only validation is enough to describe the #1215 CI state, but not to claim Julia tests ran. If a broader code change is later included, rerun the relevant suite.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-03-pr1215-low-risk-review.md`
- `docs/dev-log/check-log.d/2026-11-03-pr1215-low-risk-review.md`

The pre-existing modified Python bytecode file remains untouched.

## 5. Checks Run

- Read live PR metadata, workflow runs, and job steps for the exact SHAs recorded in the crosswalk.
- Read the complete #1500 source-review result. It found no P0–P2 correctness blocker and one P3 test-coverage suggestion. No package tests were rerun.
- Read the complete #1215 patch. Verified `docs/src/capabilities.md` and `docs/src/api-stability.md` exist on base `973bbb675dc6a10b436acf86175e089e38ec467e`.
- #1500 exact-head run 41542189558: four Ubuntu shards, source-tree check, and OS-matrix check passed.
- #1667 exact-head run 41548315716: source-tree and OS-matrix checks passed; all four Ubuntu shards were in progress.
- #1249 exact-head run 41548951015: changes job queued. Documenter run 41548951145: docs build in progress.
- #1215 run 41539026157: four Julia shard jobs concluded success through the prose-only fast path, with Julia setup/build/test steps skipped; aggregate `ci-ok` remained queued. Documenter run 41539026225: docs build and rendered-doc audit passed; docs-preview pending and deploy skipped.
- PR #1214 has 55 changed paths, all also present in #1210's 57-path diff, including `ROADMAP.md`. Its current-main docs job passed and CI remains active. PR #1210 is on stale base `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`; its recorded green runs are older. The local #1214 checkout is absent; the #1210 checkout SHA `af4c0f4d527f3e41d4dc3cac1e1fafb891996aab` differs from the PR head.
- No package tests, fits, simulations, installs, or Slurm jobs were run.
- `git diff --check`, the after-task structure check, and the prose check were run after editing.

## 6. Tests of the Tests

Not applicable. This task changed evidence documents only.

## 7a. Issue Ledger

- #1500: all listed CI checks passed; remains draft with no formal review. Owner is unconfirmed.
- #1667: source and OS-matrix checks passed; four Ubuntu shards remain active. Warning visibility remains a review follow-up.
- #1249: CI queued and docs build active. Two P2 source-review follow-ups remain.
- #1215: docs build passed; preview pending. The local branch is stale relative to the PR head; current owner is unconfirmed.
- #1214: 55-file docs/code audit PR on current main; docs passed and CI remains active. It overlaps #1210 on all 55 paths and overlaps #1215 on `ROADMAP.md`.
- #1210: 57-file docs/code audit PR on stale base, with older successful CI and docs runs. Its local checkout is stale and overlaps #1214 on 55 paths and #1215 on `ROADMAP.md`.
- No issue or PR metadata was changed.

## 8. Consistency Audit

- Matched each workflow run to the live PR head before recording it.
- Treated a skipped Julia test step as no test execution, even when the shard job concluded successfully.
- Kept internal source-review findings distinct from formal GitHub reviews.
- Kept a clean but stale worktree distinct from a current exact-head owner checkout.
- Confirmed #1215 overlaps the #1214 and #1210 documentation lanes on `ROADMAP.md`; reconcile those lanes before merging this change.
- Style assessment: 2/10 perceived AI-like style, medium confidence. Genre and coverage: this short operational report, read in full. The exact hashes and explicit skipped-test distinction make the account specific; no style change needed. Science: not applicable. Facts: checked against the workflow jobs and Git refs listed above. References: direct run links only. Provenance: self-review, this report version, 2026-11-03; no unrelated examples used.

## 9. What Did Not Go Smoothly

The PR #1215 worktree branch name matched the pull request, but its checked-out SHA did not. It is therefore not evidence that the current PR head has an active local owner.

## 10. Known Residuals

- PR ownership remains unconfirmed for #1215 and #1500; the clean #1667 checkout has no recorded handoff.
- #1215 preview remains pending. Its successful prose-only shards did not run Julia tests.
- #1215 overlaps two larger open documentation PRs; the owners have not resolved how this wording fits into their combined changes.
- #1249 and #1667 have incomplete exact-head CI at the observation time.
- No numerical R–Julia parity or interval calibration was tested.

## 11. Team Learning

Memory receipt: read the current crosswalk, repo coordination preflight, and exact PR/run evidence. The lane preflight found no competing ref missing the crosswalk path, and a scoped lease was obtained. Golden Set: not run; this was a documentation evidence update.

## 12. Cross-Product Coverage

Covers: exact-head status, read-only source review, and path-overlap analysis for six DRM parity-related PRs.

Does NOT cover: package-wide test completion, numerical parity, calibration, owner handoff, merge, release, publication, or deployment.
