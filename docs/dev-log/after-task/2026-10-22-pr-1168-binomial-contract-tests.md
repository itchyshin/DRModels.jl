# After-task: fixed-effect Binomial response and accessor contract tests

## 1. Goal

Complete the repository record for PR #1168 and make its invalid confidence-level assertion match the existing `confint` contract. The change remains limited to the new test and its two required records.

## 2. Implemented

- Added a contract test for fixed-effect `Binomial()` response refusals, zero-trial rows, and accessors for `cbind(successes, failures)` and 0/1 Bernoulli responses.
- Corrected the test's expected exception for `confint(fit; level = 1.2)` from `DomainError` to `ArgumentError`.
- Added this after-task report and one per-slice check-log fragment.
- No production source or public API behavior changed.

## 3a. Decisions and Rejected Alternatives

- Chose `ArgumentError` because `src/ci_args.jl` defines `_validate_ci_level` to reject values outside `(0, 1)` with that type, `confint` calls the validator, and neighboring invalid-level checks use `ArgumentError`.
- Rejected changing production behavior to throw `DomainError`; that would contradict the shared validator and other public methods.
- The PR body initially reported 84/84 assertions. Running the original branch source reproduced 83 passes and one failure: the test expected `DomainError`, but the validator correctly threw `ArgumentError`.
- Kept interval accessor algebra separate from statistical interval calibration. No coverage claim is made.
- Used a per-slice check-log file rather than editing the frozen aggregate log.

## 4. Files Touched

- `test/test_binomial_contract.jl`
- `docs/dev-log/after-task/2026-10-22-pr-1168-binomial-contract-tests.md`
- `docs/dev-log/check-log.d/2026-10-22-pr-1168-binomial-contract-tests.md`

## 5. Checks Run

The exact PR head `ae1b1b9456387570db0c933a923d5c1991c1afed` was merged locally with live `main` at `e4333387050549d49bf95738543e1e11bc81fa8c`, producing test candidate `4916eaa252416b47d9b99f94b6918042b9b0dd59`. The merge was clean. On that candidate, Julia 1.10.0 with 2 Julia threads and 1 BLAS thread passed the focused contract test: 84/84.

For the linear-history PR branch, the same three-file diff was applied directly to current `main`. The resulting staged tree `2a4897719b37c6da717fb4a374a235a80c7c3a5c` exactly matches the reviewed candidate tree. The focused test was rerun on that exact tree and again passed 84/84. Its raw log is `/private/tmp/pr1168-linear-candidate.log` (SHA-256 `280ef773f9c1db5d40bf2106c85315c405757582543f2becde26c0802b691d71`).

`JULIA_DEPOT_PATH=/private/tmp/pr1168-julia-depot:/Users/z3437171/.julia JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. test/test_binomial_contract.jl`

The first run could not load `ForwardDiff` because the isolated depot had not yet been instantiated. `Pkg.instantiate()` then resolved dependencies offline from the existing local depot and created an ignored temporary `Manifest.toml` in this isolated clone; it made no tracked project changes and used no network access. The passing rerun's raw log is `/private/tmp/pr1168-binomial-contract-merged-candidate.log` (SHA-256 `11df59ca53343585cbeaa747de61616a259d2335664d7a85e6ea12fc6bdfda33`).

The full `Pkg.test()` suite and R parity tests were not run. No updated-PR CI is available for this local merge candidate; CI must run after the refreshed commit is pushed.

## 6. Tests of the Tests

The PR body reports ten single-expectation mutants, each failing at least one assertion. Those mutations were not rerun here. The original invalid-level expectation was directly shown to fail, then the corrected assertion passed in the focused test.

## 7a. Issue Ledger

- Fixed: the test now expects the established `ArgumentError` for an invalid confidence level.
- Verified locally: focused Binomial contract test, 84/84 after the correction.
- Open: full Julia package suite, R parity, and CI on the refreshed PR head.
- Deferred: interval availability and statistical coverage/calibration; this PR does not establish them.

## 8. Consistency Audit

The test-only patch checks response-domain errors, zero-trial rows, independent IRLS and closed-form comparisons, accessors, simulation shape and reproducibility, and frequency-weight equivalence. The invalid-level expectation now matches the shared validator and neighboring tests. The check-log fragment has the repository's five-column row format. The shared frozen check-log remains untouched.

## 9. What Did Not Go Smoothly

The initial focused run found a mismatch between the test's exception expectation and the package's established validation error. Offline dependency resolution was needed because the isolated clone had no manifest. After correcting the test expectation, the focused run passed.

## 10. Known Residuals

The full `Pkg.test()` suite and R parity tests remain unrun. Mutation checks are PR-author-reported. CI on the refreshed remote head must complete before merge. No interval coverage or calibration evidence is included.

## 11. Team Learning

Contract tests should assert the shared validator's public error type as well as its message. An invalid-input test can uncover a test defect even when the implementation contract is already consistent.

## 12. Cross-Product Coverage

Covers: fixed-effect Binomial response-domain and accessor behavior for cbind trial counts and 0/1 Bernoulli responses, including the invalid confidence-level error type.

Does not cover: the full Julia package suite, R parity, random-effect Binomial routes, other response families, interval availability, or statistical interval coverage and calibration.
