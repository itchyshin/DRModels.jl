# After-task: BetaBinomial and truncated NB2 accessor contract tests (#1157)

## 1. Goal

Record the test-only contract for `BetaBinomial()` and `TruncatedNegBinomial2()` without changing either family's implementation or broadening the tested behavior.

## 2. Implemented

- The PR adds `test/test_betabinomial_tnb2_contract.jl`, covering fixed-effect accessors, parameter-scale interpretation, refusals, residuals, Wald intervals, information criteria, simulation support, and independent likelihood calculations for the two families.
- This continuation adds this after-task report and the matching per-slice check-log row. No source files or unrelated tests were changed.
- Current `main` was merged into the original PR head in the isolated clone. The merge had no conflicts and preserves the PR commit as an ancestor, so the branch update remains fast-forwardable.

## 3a. Decisions and Rejected Alternatives

The test remains fixed-effect-only. Random-effect behavior for `TruncatedNegBinomial2()` remains outside this slice and tracked by issue #1122. The zero-trial BetaBinomial refusal expectation remains explicitly `@test_broken`, matching current behavior rather than implying that it is fixed.

The historical CI timing failure is in the unchanged `test/test_tnb2_nan.jl` timing-ratio guard. It is recorded as a flaky timing result, not repaired by changing unrelated assertions or weakening that guard in this PR.

## 4. Files Touched

- `docs/dev-log/after-task/2026-10-22-pr1157-test-contract.md`
- `docs/dev-log/check-log.d/2026-10-22-pr1157-test-contract.md`
- PR's existing `test/test_betabinomial_tnb2_contract.jl` (unchanged in this continuation)

## 5. Checks Run

On Julia 1.10.0 with `OPENBLAS_NUM_THREADS=1` and `JULIA_NUM_THREADS=4`, dependencies were instantiated in the isolated clone without changing `Project.toml`. The focused command

```sh
julia --project -e 'using Test; using DRModels; include("test/test_betabinomial_tnb2_contract.jl")'
```

passed the BetaBinomial set (33 pass, 1 expected broken) and truncated NB2 set (29 pass). The temporary `Manifest.toml` is ignored and is not part of the PR diff. `git diff --check github/main...HEAD` passed before adding these records; it will be rerun on the final candidate.

The historical PR run had 7 of 8 shard jobs pass. The one failure was the timing-ratio check in unchanged `test/test_tnb2_nan.jl`. The Documenter build passed; its preview job was cancelled. Those are historical CI results, not a claim about checks on the updated head.

## 6. Tests of the Tests

The newly added contract file was executed directly on the merged candidate. Its one broken assertion is the documented zero-trial refusal gap; all other assertions passed. The full package suite was not run in this continuation.

## 7a. Issue Ledger

The contract tracks issue #1122. It does not resolve the separately identified zero-trial refusal gap or add random-effect support.

## 8. Consistency Audit

The recorded scope matches the test header: fixed effects only, with no duplicated recovery tests and no random-effect claims. Test output separates passing checks from the expected broken assertion. Historical CI is distinguished from new local verification.

## 9. What Did Not Go Smoothly

The isolated clone initially lacked a resolved Manifest. `Pkg.instantiate()` resolved the project's declared dependencies locally. A first unprivileged Julia load hit a cache-permission error; the focused test then completed successfully after the dependencies were available. The lane-lease registry lock was busy, so the preflight's no-active-lane result and the isolated clone were used to keep the edits narrowly scoped.

## 10. Known Residuals

The focused test does not establish that the full suite is green. The historical timing guard failure needs a current exact-head CI result before any merge decision. The PR's updated CI has not yet been observed.

## 11. Team Learning

For test-only PRs, an explicit expected-broken result and exact denominator make the contract clearer than silently weakening an assertion. When updating a branch, merging current `main` into the original PR head preserves its ancestry and permits a normal fast-forward push.

## 12. Cross-Product Coverage

This slice covers the Julia fixed-effect accessor contract for `BetaBinomial()` and `TruncatedNegBinomial2()`. It does not cover R parity, random effects, implementation changes, the full package suite, or statistical interval calibration.
