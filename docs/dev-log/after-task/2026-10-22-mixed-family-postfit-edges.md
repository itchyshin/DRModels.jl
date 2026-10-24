# After Task: Mixed-family post-fit accessor edge contracts

## Goal

Add deterministic regression coverage for mixed-family post-fit accessors and edge cases in PR #1158.

## Implemented

The PR adds one test file covering coefficient tables, AIC/BIC warnings and guards, stored-design fitted values, summary output, and confidence-interval printing. The test uses a fixed random seed and one Gaussian/Poisson mixed-family fit. It adds no package source or public behavior.

## Files Changed

- `test/test_mixed_family_postfit_edges.jl`
- `docs/dev-log/check-log.d/2026-10-22-mixed-family-postfit-edges.md`
- `docs/dev-log/after-task/2026-10-22-mixed-family-postfit-edges.md`

## Checks Run

- PR head `81d8963354a56455eda16d83c3d143e6fa2af5a8` was 44 commits behind current main `b8dbf64e9065724bc359f139ff914d708e788bb6`; the branch was updated with current main without conflicts. The patch adds only the test file, so no changed-path overlap was present.
- On the original PR head, CI run 39355569874 reports all Julia 1 and 1.10 shards plus `ci-ok` and `changes` successful. The Documenter build succeeded; `docs-preview` was pending when checked. These results predate the main refresh and are not evidence for the refreshed head.
- A local focused run was attempted with Julia 1.10. The default depot was not writable for the compile lock. Retrying with a writable `/private/tmp` depot then stopped because ForwardDiff is unavailable in the installed depots. The test therefore has not been independently rerun against current main. Fresh PR CI is required before merge.
- `git diff --check` passed after the documentation additions.

## Tests Of The Tests

The test fixes the random seed, verifies both expected values and refusal/warning paths, and exercises invalid `nobs`, absent or partial design matrices, a likelihood sentinel, and non-finite intervals. It checks the intended accessors directly after one mixed-family fit. It does not measure recovery, interval coverage, or performance.

## Consistency Audit

The patch changes tests and their records only. The existing public API, docstrings, worked user examples, and package behavior are unchanged. The test file uses the current `mf_*` accessors and does not assert a new capability.

## What Did Not Go Smoothly

The available Julia depot lacks ForwardDiff, and the protected default depot could not create a compile lock. No local test result is claimed.

## Team Learning

For a test-only PR, record the exact tested source and separate historical CI from checks on a refreshed main. Keep local dependency/cache failures visible rather than treating them as test failures or successful verification.

## Design Docs, Worked Example, and NEWS

No design or user documentation change is warranted because this PR tests existing behavior and adds no user-facing feature. The regression test is the executable example for these internal edge cases. No NEWS fragment is added for test-only coverage.

## GitHub Issue Maintenance

No linked issue was identified in the PR metadata; the issue tracker is unchanged.

## Known Limitations

This test addition is not evidence of statistical recovery, confidence-interval calibration, or broad mixed-family model validation. The refreshed PR head still needs its full required CI checks, including docs preview, before it can be considered for merge.

## Next Actions

Wait for all checks on the refreshed exact PR head to complete successfully, then review mergeability and draft status again.
