# After-task: SkewNormal and TruncatedPoisson accessor contract tests (#1149)

## 1. Goal

Record the fixed-effect accessor contract tests for `SkewNormal()` and `TruncatedPoisson()` without changing either implementation.

## 2. Implemented

- Added the required after-task report and collision-free per-slice check-log entry to the existing test-only PR branch.
- The PR's existing `test/test_skewnormal_truncpoisson_contract.jl` remains unchanged. It covers 34 SkewNormal assertions and 27 TruncatedPoisson assertions.
- No source, formula, estimator, or public API file changed.

## 3a. Decisions and Rejected Alternatives

The slice pins current fixed-effect behavior for SkewNormal and non-hurdle TruncatedPoisson accessors. It does not make claims about random effects, the separate near-zero SkewNormal Wald warning, or interval calibration. Tests of confidence-interval arithmetic establish the accessor contract only; they do not validate coverage.

The test was run using the already-resolved docs environment, with `DRModels` developed from the exact PR worktree. The package-root environment did not have `ForwardDiff` installed, so its initial load failed before any assertion ran. No dependencies or manifests were added to the PR branch.

## 4. Files Touched

- `docs/dev-log/after-task/2026-10-23-skewnormal-truncpoisson-contract.md`
- `docs/dev-log/check-log.d/2026-10-23-skewnormal-truncpoisson-contract.md`
- Existing PR test file, unchanged.

## 5. Checks Run

The test source was PR head `1351a78693296e046f26566208e1aac2fe75da77`, based on current `main` `898b2e11ae0d1fddc1ce71a7ba93aaa50bc45957`. The local recheck loaded `DRModels` from `/Users/z3437171/local-scratch/lanes/red/DRM.jl-1149/src/DRModels.jl` at that exact source revision.

On Julia 1.10.0, with `JULIA_DEPOT_PATH=/private/tmp/drmodels-pr1149-depot:/Users/z3437171/.julia`, `JULIA_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`, the focused test file passed in the isolated docs environment:

```sh
JULIA_DEPOT_PATH=/private/tmp/drmodels-pr1149-depot:/Users/z3437171/.julia JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=/private/tmp/drmodels-docs-current-20261023.eVoHCz/docs -e 'using DRModels; using Test; println("DRModels source: ", pathof(DRModels)); include("/Users/z3437171/local-scratch/lanes/red/DRM.jl-1149/test/test_skewnormal_truncpoisson_contract.jl")'
```

Results: SkewNormal accessor contract, 34/34 passed; TruncatedPoisson accessor contract, 27/27 passed. The temporary docs environment and its Manifest are outside the PR worktree.

`julia tools/build_check_log.jl --check` exits 1 on existing multi-line files in `docs/dev-log/check-log.d/`. I confirmed representative failures are already present on `origin/main`; this slice's new entry is one well-formed table row. No historical records or validator code were changed.

At the time of this report, GitHub CI run `39895953264` and Documenter run `39895953252` were queued on the exact PR head. The PR remains a draft. Reviews and inline review threads were empty. The full package suite was not run.

## 6. Tests of the Tests

The test file ran against the exact PR source and exercised both family-specific accessor testsets. All 61 assertions passed. No interval-coverage simulation or family recovery campaign was run.

## 7a. Issue Ledger

The test-only PR references issues #1123 and #1060. It does not close either issue or address the separately tracked near-zero SkewNormal Wald warning.

## 8. Consistency Audit

The report matches the test file's fixed-effect scope and measured assertion counts. It distinguishes locally passing focused tests from queued package CI, and accessor arithmetic from statistical interval calibration.

## 9. What Did Not Go Smoothly

The package-root environment initially failed before assertions because `ForwardDiff` was absent from the active Julia depots. A later rerun using the docs environment first hit a read-only home-depot compile-lock error, also before assertions. Using a temporary writable depot then completed the test successfully. Neither attempt changed the PR source or project files.

## 10. Known Residuals

- Exact-head CI and Documenter remain queued; their results are not known.
- The PR remains draft and is not merge-ready until its required repository checks and review gates are satisfied.
- This focused test does not validate interval calibration, coverage, random-effect routes, or the full package suite.

## 11. Team Learning

When a test-only branch has no root Manifest, reuse its resolved docs environment in a temporary depot and develop the exact worktree there. Record the loaded source path and keep generated environment files outside the PR.

## 12. Cross-Product Coverage

This slice covers only the Julia fixed-effect accessor contracts for `SkewNormal()` and `TruncatedPoisson()`. It does not establish R parity, random-effect behavior, coverage validity, or general DRM readiness.
