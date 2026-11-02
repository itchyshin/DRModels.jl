# After-task: Gamma and Beta fixed-effect contract tests (#1159)

## 1. Goal

Refresh the narrow Gamma/Beta accessor-contract test PR onto the specified live `main` while preserving its test-only scope.

## 2. Implemented

- Rebased the single test-file change onto `main` at `b8dbf64e9065724bc359f139ff914d708e788bb6`.
- Added the per-slice check-log entry and this after-task report.
- No model source or public API changed.

## 3a. Decisions and Rejected Alternatives

- Kept the two Beta `@test_broken` assertions: current behavior returns the generic support error rather than advice mentioning `ZeroOneBeta`; changing this test to claim otherwise would be false.
- Did not add a NEWS fragment because the slice adds tests only and changes no user-facing behavior.
- Assumptions made without asking: none.

## 4. Files Touched

- `test/test_gamma_beta_contract.jl` (existing PR file; unchanged during refresh).
- `docs/dev-log/check-log.d/2026-10-22-1159-gamma-beta-contract.md` (new).
- `docs/dev-log/after-task/2026-10-22-1159-gamma-beta-contract.md` (new).

## 5. Checks Run

- `git fetch origin main refs/pull/1159/head:refs/remotes/origin/pr/1159`: verified main `b8dbf64e9065724bc359f139ff914d708e788bb6` and original PR head `41f76aed9158d642a1db3801e64dad846040b9e5` before changes.
- `git diff --check`: PASS on an isolated reconstruction based at `b8dbf64e9065724bc359f139ff914d708e788bb6`, using all three files fetched from the exact PR head `f24f22a96315488cc35a1acd7c86b53758b216fe`. Their Git blob IDs matched the live PR files; the check therefore included the test, check-log, and after-task files.
- Focused Julia test on the rebased source: 109 passed and 2 declared broken assertions; details below.
- `julia tools/build_check_log.jl`: exited 0; the renderer warns about pre-existing malformed legacy entries in other check-log files. The new row uses the required one-row format.
- `check-after-task.R`: structural check passed. The broader closeout compiler is blocked by unrelated pre-existing `UNMET` gates in the Shinichi brain repository's unlazy ledger; none belongs to this PR.

## 6. Tests of the Tests

The assertions compare fitted accessors and likelihoods to response-scale formulas, closed-form Gamma/Beta densities, interval arithmetic, seeded simulation behavior, and explicit refusal messages. The two `@test_broken` assertions mark known missing `ZeroOneBeta` guidance. No production mutation test or red-on-main run was performed in this refresh; the PR is test-only.

## 7a. Issue Ledger

- No issue was closed by this test-only PR. The PR references the test-coverage campaign.
- The generic Beta support message for exact 0/1 remains a visible test gap, not a behavior change in this slice.

## 8. Consistency Audit

The refreshed diff remains one added test file plus collision-free check-log and after-task records. It does not alter family fitters, shared runner registration, parity fixtures, or prediction code. The broader suite and R parity were not run.

## 9. What Did Not Go Smoothly

The first focused test invocation could not resolve the local `DRModels` package, and the shared Julia depot was not writable. A temporary test environment and compiled cache were created inside `/private/tmp`; package resolution required network access. The targeted test then completed.

## 10. Known Residuals

- Full `Pkg.test()` and R parity were not run for the refreshed candidate. Exact-head Julia CI ran the Documenter and docs-preview jobs successfully; this is not a stable documentation deployment.
- The two Beta advice assertions remain broken by design until the user-facing refusal changes.
- Exact existing PR head `f24f22a96315488cc35a1acd7c86b53758b216fe` passed `ci-ok`, Julia 1 and Julia 1.10 shards, Documenter, and docs-preview; `docs-deploy` and `scaling-sweep` were skipped. This report-only update will create a new head, which requires fresh exact-head checks before merge.

## 11. Team Learning

An isolated test PR may need a local package-development entry in its temporary Julia test environment. Keep that dependency and compiled cache outside the shared checkout, and do not commit the temporary Manifest.

## 12. Cross-Product Coverage

- Covers: fixed-effect Gamma and Beta refusal/accessor contracts in Julia tests.
- Does NOT cover: random-effect or structured Gamma/Beta routes, R implementation, R-to-Julia parity, interval calibration, the full package suite, or production source behavior changes.

Memory receipt: the repository context graph map was consulted. The repository lane-preflight script was not present in this checkout; the parent task authorized this isolated PR refresh. No memory update was requested.

Golden Set: not run; this test-only refresh did not touch a known-mistake class.
