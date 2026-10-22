# After Task: Tweedie contract test suite (#1153)

## 1. Goal

Record the test behavior covered by PR #1153 and verify the new test file against current live main.

## 2. Implemented

The test-only PR adds `test/test_tweedie_contract.jl`. It checks the Tweedie zero-mass formula and density normalization, unsupported-input refusals, a fixed-effect fit's accessors, and two degenerate-response outcomes. The test runner discovers the new file automatically.

## 3a. Decisions and Rejected Alternatives

Kept the existing test source unchanged. The test verifies deterministic density, refusal, and accessor behavior; it does not add implementation, claim statistical recovery, or assess interval coverage. No NEWS fragment is needed for a test-only change.

## 4. Files Touched

- Added `test/test_tweedie_contract.jl` in the original PR; SHA-256 `45b5378ce6860a8fdfa85f1813eaf6b9c117215006c89a41a55107c5420becba`, unchanged from source head `e6c766b5dd2f1ace9252160edd781e81c6585117`.
- Added `docs/dev-log/check-log.d/2026-10-22-pr1153-tweedie-contract-tests.md`.
- Added this after-task report.
- Merged current live main `b8dbf64e9065724bc359f139ff914d708e788bb6` into the isolated PR branch without conflicts.

## 5. Checks Run

- Offline environment preparation: `JULIA_DEPOT_PATH=/private/tmp/drmodels-pr1153-depot:/Users/z3437171/.julia JULIA_PKG_OFFLINE=true OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=test -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'` completed using the existing cache; no package sources were downloaded, and no tracked project file changed.
- Focused test: `JULIA_DEPOT_PATH=/private/tmp/drmodels-pr1153-depot:/Users/z3437171/.julia OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=test -e 'using DRModels; include("test/test_tweedie_contract.jl")'`: PASS, 54/54 in 15.5 seconds on Julia 1.10.0.
- `julia tools/build_check_log.jl`: rendered this fragment as the expected row (line 454 in the generated preview). The builder also reported malformed legacy files elsewhere in `check-log.d/`; those files are outside this PR and were left untouched.
- `git diff --check`: PASS.
- The first focused-test attempt was blocked because the test environment had no manifest; a run with the root project was blocked by the sandbox's read-only default Julia cache. The successful run used a temporary writable depot under `/private/tmp` and the existing package cache.
- Full `Pkg.test()` and a fresh Documenter build have not been run locally. Fresh PR CI and Documenter results on the updated head are required before promotion.

## 6. Tests of the Tests

The zero-response log density is checked against its closed-form atom, and quadrature checks atom plus continuous density normalization for three power parameters. Refusal tests pin error messages for unsupported response and model shapes. The accessor test exercises prediction, parameter prediction, fitted values, simulation, residuals, covariance, standard errors, confidence intervals, model criteria, and deviance on one fixed-effect fit. Degenerate all-zero and one-positive response cases pin the current warnings and fit state.

## 7a. Issue Ledger

Open issue #992 documents a separate dispersion-clamp caveat for Tweedie and other families. This test does not evaluate that clamp or the documentation claim; issue #992 remains open and unchanged. No other issue was linked or changed.

## 8. Consistency Audit

`test/runtests.jl` discovers `test_*.jl` files by directory listing, so no registration edit is needed. The test-only scope changes no public API, formula grammar, documentation page, NEWS, or capability claim. The active `tweedie-aghq` worktree modifies `src/tweedie.jl`, `test/runtests.jl`, and AGHQ-specific tests; it does not touch this PR's new test path. The broader general-SD lane also has no file overlap. No Documenter page changed.

## 9. What Did Not Go Smoothly

The isolated clone inherited an unpopulated checkout index, then was switched to the exact PR head from cached objects. The first local run lacked a resolved test environment. Resolving dependencies offline in `/private/tmp` exposed cached package sources and allowed the focused test to pass without changing tracked project files. Julia emitted a non-fatal upstream JET `@lookup` deprecation warning during precompilation.

## 10. Known Residuals

This is focused contract-test evidence only. It does not establish recovery, interval calibration, full-suite health, or Tweedie dispersion-clamp behavior. Full package tests and Documenter must pass on the updated exact head before the PR is promoted.

## 11. Team Learning

Julia test-only PRs still need the per-slice check-log fragment and after-task record. An offline temporary depot can run a focused test when the default compiled-cache path is unavailable and package sources are already cached.

## 12. Cross-Product Coverage

This change covers the Julia fixed-effect Tweedie route named in the test file. It does NOT cover drmTMB parity, random-effect Tweedie routes, dispersion-clamp limits, recovery, interval calibration, or full package acceptance.
