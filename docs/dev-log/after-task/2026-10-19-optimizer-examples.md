# After-task: README optimizer-selection example

## 1. Goal

Make the existing DRModels optimizer choices easy to discover from the README and direct readers to the full route qualifications.

## 2. Implemented

Added a compact Gaussian location–scale example requesting `:bfgs` and `:lbfgs`, with `control_info(fit_bfgs)` to inspect the requested and actual method. The text states that the data object has numeric `y` and `x` columns, that `:default` retains the current optimizer and restart policy, and that selection applies to supported primary ML routes. The convergence guide remains the detailed reference.

## 3. Decisions and rejected alternatives

Kept the existing convergence guide as the single detailed explanation; duplicating its starts, restarts, and refusal conditions in the README would make the entry point harder to scan. No fitting campaign or product change was in scope.

## 4. Files touched

- `README.md`
- `docs/dev-log/check-log.d/2026-10-19-optimizer-examples.md`
- `docs/dev-log/after-task/2026-10-19-optimizer-examples.md`

## 5. Checks run

`git diff --check` passed. Manually checked the snippet against the existing README `bf`/`drm` usage and the convergence guide's documented `DRMControl(optimizer = :bfgs | :lbfgs)` and `control_info` fields. Confirmed the relative guide link resolves in the repository. No numerical fits, tests, or full Documenter build were run because the change only adds a README pointer and the docs build executes examples.

## 6. Tests of the tests

Not applicable: no tests were added or changed, and no package code changed.

## 7. Issue ledger

No issue was named for this documentation slice. The optimizer-selector implementation and detailed route qualifications pre-exist this branch.

## 8. Consistency audit

The README uses the same `:bfgs`, `:lbfgs`, `:default`, `DRMControl`, and `control_info` names as the convergence guide and package API. The text limits the claim to supported primary ML routes and links the reader to route-level caveats.

## 9. What did not go smoothly

The isolated worktree initially pointed at the optimizer-selector branch tip. It was cleanly switched to a new branch from the freshly fetched `origin/main` before edits. No unrelated files were changed.

## 10. Known residuals

The example was not executed in this slice. Its API shape is already exercised by the selector tests and the detailed convergence guide; this report does not claim a fresh fit or full docs-build result.

## 11. Team learning

A short README entry is enough for discovery when the convergence article already owns method support boundaries and inference caveats. The lease was recorded against the canonical `DRM.jl` repository key before edits.

## 12. Cross-product coverage

Covers README discovery of explicit optimizer selection for supported primary ML routes. This slice does not change or revalidate optimizer behavior, inference, parameter recovery, performance, or the full documentation build.
