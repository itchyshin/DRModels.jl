## 1. Goal

Reconcile the Julia capability-status ledger with current `origin/main` source citations and public capability labels.

## 2. Implemented

Updated stale source line references, aligned the cross-family bivariate row with its recorded experimental status, refreshed the 49-row status snapshot, and removed obsolete statements that the public capabilities page is stale. Kept interval calibration and parity claims outside the scope of this documentation-only change.

## 3a. Decisions and Rejected Alternatives

Kept cross-family bivariate support experimental under the recorded A11 decision. Rejected adopting the rename and capability-status branches after review because their diffs remove unrelated evidence or change status wording. Used a clean branch from current cached `origin/main` so those lane changes remain untouched.

## 4. Files Touched

- `docs/design/capability-status.md`
- `docs/dev-log/check-log.d/2026-11-02-parity-crosswalk-reconciliation.md`
- `docs/dev-log/after-task/2026-11-02-parity-crosswalk-reconciliation.md`

## 5. Checks Run

`python3 tools/check_capability_citations.py` passed after correcting three stale citations; 17 citations were reported as skipped because the checker could not resolve them precisely. `git diff --check` passed. A direct status-table parse confirmed 49 rows with counts 43 implemented, 2 rejected, 1 planned, 2 experimental, and 1 missing. No Julia tests or Documenter build were run because this slice changes only design documentation.

## 6. Tests of the Tests

Negative-control check: the capability citation checker found stale source locations before the corrections and passed afterward. This validates citation freshness only; it does not test model behavior or capability correctness.

## 7a. Issue Ledger

Fixed stale line citations, outdated public-page claims, and the cross-family status mismatch. Deferred Julia full-suite status, rendered-site review, R–Julia parity qualification, and interval calibration to their separate workstreams.

## 8. Consistency Audit

Compared the ledger against current `origin/main` source include/export locations and `docs/src/capabilities.md`. The capability totals and row labels agree with the reviewed snapshot. The public page remains a summary; this change does not claim that every route has statistical validation.

## 9. What Did Not Go Smoothly

The shared `closeout.py new` helper initially wrote its skeleton under the brain path instead of this worktree. The file was moved into the intended repository, the unintended brain-path copy was removed, and the report’s generated paths were replaced with the actual touched files. The after-task validator was first invoked with `python3`; it is an R script and must be run with `Rscript`.

## 10. Known Residuals

The citation checker still skips 17 items it cannot resolve precisely. The Julia test suite and Documenter site were not run. This ledger reconciliation does not establish full R–Julia parity, interval availability, calibration, release readiness, or deployment. A live PR scan found PR #1245 also edits `docs/design/capability-status.md` and contains a broader evidence update. At the check on 2026-11-02, its head was `7f6e335fd46c479b874d10525ad4cf3d74b59948` and its Julia shards, Documenter build, and preview were green; no formal review was recorded. The related local worktree `/Users/z3437171/.codex/worktrees/parity-ledger-aliases/DRM.jl` was on a different branch head (`0dde1f977`) with 22 modified tracked files and 54 untracked entries. Three other local worktrees also had capability-page edits: `capability-status-reconcile`, `cross-family-interval-language`, and `drm-finish-crosswalk`. This branch has no PR; keep it separate until its citation corrections are compared with the active work, to avoid a conflicting or scope-losing submission.

## 11. Team Learning

Memory receipt: repository route and relevant brain context were checked. The route script produced no `LOAD-FIRST` manifest. Repository instructions, Graft index/context, ownership constraints, and current-main source references were reviewed; no concurrent worktree was edited.

Golden Set: not applicable to a documentation-only reconciliation.

## 12. Cross-Product Coverage

Covers: current-main Julia capability-status wording and source citation reconciliation.

Does NOT cover: model tests, the full Julia suite, rendered-site visual review, R–Julia parity, interval calibration, PR creation/updates, merges, releases, or publication.
