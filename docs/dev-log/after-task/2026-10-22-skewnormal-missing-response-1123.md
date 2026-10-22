# After-task: SkewNormal missing-response row alignment (#1123)

## 1. Goal

Keep the existing SkewNormal missing-response behavior and regression test, move its changelog entry to the repository's conflict-free news-fragment format, and add the required per-slice records.

## 2. Implemented

Removed only PR #1165's direct entry from `NEWS.md` and added the same scoped claim as `news/1123-skewnormal-missing-response.md`. Added this after-task report and a one-row entry in `docs/dev-log/check-log.d/`. The implementation and regression-test hunks from PR #1165 were preserved unchanged.

## 3a. Decisions and rejected alternatives

Kept the claim narrow: a missing or NaN response is dropped with a warning, the fit uses observed rows, and fitted parameters/scales are rebuilt for the original rows. No interval-calibration or broader family-readiness claim is made. Used a separate news fragment instead of retaining a shared-file edit in `NEWS.md`.

## 4. Files touched

- `NEWS.md` (removed this PR's direct entry)
- `news/1123-skewnormal-missing-response.md`
- `docs/dev-log/check-log.d/2026-10-22-1123-skewnormal-missing-response.md`
- `docs/dev-log/after-task/2026-10-22-skewnormal-missing-response-1123.md`

## 5. Checks run

On the original PR head `abe4c4e6f0493610ccf79db5e5984e8f9eb447ee`, Julia 1.12.6 with `--check-bounds=yes` ran `test/test_1089_missing_rows_dropped.jl`: 51/51 shared missing-row assertions and 7/7 SkewNormal assertions passed. I reran that focused test after merging live main; the tested source candidate was `368fb8b2dd1fbc2addf19b8fac07ade89cecde4f`, with the same 51/51 and 7/7 passing. At the original pre-refresh head, GitHub Actions passed all eight Julia matrix shards, `ci-ok`, Documenter `docs`, and `docs-preview`. The refreshed branch includes live-main commits `b8dbf64e9065724bc359f139ff914d708e788bb6` and `08348d8e6`, which add two independent test files; full CI has not yet rerun on the refreshed head.

## 6. Tests of the tests

No test source was changed in this records-only update. The existing SkewNormal assertions check the warning, observed-row `nobs`, likelihood and location agreement with a hand-filtered reference fit, and full-row scale reconstruction.

## 7a. Issue ledger

Addresses the remaining row-alignment behavior in #1123. PR #1165 retains its existing closure reference.

## 8. Consistency audit

The fragment describes the behavior exercised by the existing test and implemented in `src/skewnormal.jl`. It does not claim interval validity, coverage, or support for other missing-data routes.

## 9. What did not go smoothly

Live main advanced by two commits after the PR's recorded base. They were merged cleanly in the isolated PR clone and do not touch the SkewNormal implementation or its regression-test file. `NEWS.md` is also edited by open PRs #1145 and #1146, so this PR's own entry was moved to its unique fragment path.

## 10. Known residuals

The refreshed head needs its own CI and Documenter run. No merge was performed. No interval-calibration or coverage study was run; interval capability is not interval validation.

## 11. Team learning

When a PR updates an existing `NEWS.md` entry during parallel work, convert that PR's entry to a unique fragment before refreshing against main. Preserve prior test evidence with its exact tested head.

## 12. Cross-product coverage

This slice only covers the Julia SkewNormal missing-response path. It does not establish R parity, bridge support, CMR readiness, or interval coverage.
