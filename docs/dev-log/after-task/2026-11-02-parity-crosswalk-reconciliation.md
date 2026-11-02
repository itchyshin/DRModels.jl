## 1. Goal

Reconcile the Julia capability-status ledger with current-main source evidence while preserving explicit route limits and the existing cross-language parity boundary.

## 2. Implemented

Updated stale source references, retained the explicit bivariate ordinary-random-effect refusal and the A11 experimental boundary for cross-family bivariate fits, and added two source-and-test-backed rows: ordinary non-Gaussian random intercepts using explicit `marginal = :Laplace`, and coupled NB2 location-scale random intercepts. The snapshot now records 51 rows. The new rows state that their implementation tests do not establish interval calibration or general R–Julia parity.

## 3a. Decisions and Rejected Alternatives

Kept cross-family bivariate support experimental under the recorded A11 decision and retained the precise bivariate Gaussian refusal. Reviewed four refs that carry alternate versions of this page. Preserved the bounded ordinary-Laplace and coupled NB2 claims, but did not adopt stale package-name changes, unsupported receipt claims, or status downgrades. The working branch is three commits ahead and two behind current `origin/main`; the relevant implementation and test files were compared directly with `origin/main` before adding the claims.

Assumption made without asking: current `origin/main` is the comparison target for this crosswalk. If the intended target is a frozen historical candidate, its source pin and status counts need a separate snapshot.

## 4. Files Touched

- `docs/design/capability-status.md`
- `docs/design/capability-status.assessment.json`
- `docs/dev-log/check-log.d/2026-11-02-parity-crosswalk-reconciliation.md`
- `docs/dev-log/after-task/2026-11-02-parity-crosswalk-reconciliation.md`
- `docs/dev-log/after-task/2026-11-02-parity-crosswalk-reconciliation.assessment.json`

## 5. Checks Run

`python3 tools/check_capability_citations.py` passed: 3 citations verified and 17 skipped as not precisely resolvable. `git diff --check` and the after-task structure check passed. A status-table count that handles escaped pipes confirmed 51 rows: 45 implemented, 2 rejected, 1 planned, 2 experimental, and 1 missing. Exact-draft assessment checks recorded both prose reviews; `slop_check.py` returned zero findings for each edited prose file. No Julia tests or Documenter build were run because this slice changes only design documentation.

## 6. Tests of the Tests

The citation checker validates only resolvable source locations. The two added rows were checked against `src/ordinary_laplace.jl`, `src/locscale_frontend.jl`, `test/test_ordinary_laplace.jl`, and `test/test_locscale_frontend.jl` in current `origin/main` (`47a081ba035f9bf3fd7e42d3841ae8a288bf2a14`); those source and test files are byte-identical to the candidate branch. This supports route existence and the stated implementation-test scope, not calibration.

## 7a. Issue Ledger

Fixed stale citations and added the two bounded capability rows. Deferred Julia full-suite status, rendered-site review, general R–Julia parity qualification, and interval calibration to their separate workstreams.

## 8. Consistency Audit

Compared the added rows with current-main source, tests, and the public capability summary. The exact bivariate refusal remains distinct from supported structured q4 routes; cross-family bivariate remains experimental. The table and snapshot counts agree. The public page remains a summary and this edit does not claim statistical validation for every route.

## 9. What Did Not Go Smoothly

The route script returned no `LOAD-FIRST` manifest for this worktree, and the `graft` command and checked-in graph index were unavailable. I used exact source and test paths from the reviewed diffs instead. The after-task validator is an R script and must be run with `Rscript`.

## 10. Known Residuals

The citation checker still skips 17 items it cannot resolve precisely. I did not run the Julia suite or Documenter for this documentation-only candidate. This edit does not establish full R–Julia parity, interval availability, calibration, release readiness, or deployment.

PR #1245 also changes this page. At the 2026-11-02 09:10 UTC refresh:

- Its head is `0dde1f977f6c70495230b705a7d0f13daf74dfde`, 28 commits ahead and 5 behind current main `47a081ba035f9bf3fd7e42d3841ae8a288bf2a14`.
- CI run `41406599132` is on that exact head. Shards 1, 3, and 4 plus the changes job passed; shard 2 remains in progress, and scaling was skipped. Documenter run `41406599136` passed docs and preview checks; stable deployment was skipped.
- No formal review is recorded. The matching local worktree contains extensive dirty source and evidence changes.

This candidate is local commit `2d1f68d87922f681049e5c2d25a0a9a25a5fd110` on `codex/drm-parity-crosswalk-20261102`. I did not push it because it changes the same page as the active #1245 work and could conflict with its owner’s uncommitted reconciliation. The handoff gate confirmed one unpushed commit on this branch; resume by comparing this commit with the final #1245 merge candidate, then either drop overlapping claims or rebase and open a non-duplicative PR after ownership clears.

The approved `meta_V` accessor target is heterogeneity SD `tau`, but current source at the reviewed baseline still returns total per-study SD `sqrt(vᵢ + tau²)`. The candidate receipts for the proposed accessor and `simulate(newdata)` changes were not present in the reviewed ref, so neither implementation nor parity is claimed here. No implementation files were changed.

## 11. Team Learning

Memory receipt: relevant parity decision D-203 and the project route were checked. The route script produced no `LOAD-FIRST` manifest. Repository instructions and four page diffs were reviewed; other refs and worktrees were left untouched.

Golden Set: not applicable to a documentation-only reconciliation.

## 12. Cross-Product Coverage

Covers: current-main Julia capability-status wording, two narrowly scoped native routes, source/test citations, and documented parity limits.

Does NOT cover: model tests, the full Julia suite, rendered-site visual review, R–Julia parity, interval calibration, PR creation/updates, merges, releases, or publication.
