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

PR #1245 also changes this page. At the 2026-11-02 09:21 UTC refresh:

- Its head is `0dde1f977f6c70495230b705a7d0f13daf74dfde`, 28 commits ahead and 5 behind current main `47a081ba035f9bf3fd7e42d3841ae8a288bf2a14`.
- CI run `41406599132` is on that exact head. Shards 1, 3, and 4 plus the changes job passed; shard 2 remains in progress, and scaling was skipped. Documenter run `41406599136` passed docs and preview checks; stable deployment was skipped.
- No formal review is recorded. The matching local worktree contains extensive dirty source and evidence changes.
- Its PR description is stale: it says this head is not the PR head and has no checks, while current PR metadata names this exact SHA and run `41406599132` is on it.
- I fetched the exact PR head for a page-level comparison. It already contains both ordinary-Laplace and coupled-NB2 rows, so this branch would duplicate those entries. The coupled-NB2 paragraph on #1245 names standard errors and Wald output but does not say that its implementation test establishes neither interval coverage nor R–Julia parity; this branch states that limit. The #1245 page is 829 lines, while this branch is 427 lines; the wider PR diff also changes substantial historical material, so do not replace either page wholesale without owner review.

This candidate is now commit `1419274288e209cdf6d9994064e7e8815f7c34b9` on `codex/drm-parity-crosswalk-20261102`. At 2026-11-02 09:12 UTC I verified that the remote branch still pointed to its parent `8866d88656733d407a62c9a9c7d0126ef36fd7aa`, confirmed the push would fast-forward, and pushed the candidate. The branch has no PR. I kept it separate from #1245 because that PR still changes the same page and its owner’s worktree is dirty. Resume by comparing this branch with #1245’s final merge candidate, then either drop overlapping claims or open a non-duplicative PR after ownership clears. The push does not trigger CI or stable documentation deployment; both workflows run on pull requests, while Documenter push jobs are restricted to `main` and version tags.

The approved `meta_V` accessor target is heterogeneity SD `tau`, but current source at the reviewed baseline still returns total per-study SD `sqrt(vᵢ + tau²)`. The candidate receipts for the proposed accessor and `simulate(newdata)` changes were not present in the reviewed ref, so neither implementation nor parity is claimed here. No implementation files were changed.

### Cross-repository cleanup status at 2026-11-02 09:21 UTC

- drmTMB PR #1662 is open at `01a2c2a49a1450cc79e9d35a138028ec1a27c720`. Its test-only fix passed all six jobs in R-CMD-check run `41350326046`; the focused Mac test passed locally, and an internal review found no blocking issue. GitHub has no formal review. The exact-head checks do not include a full Mac R CMD check, and no merge was made.
- DRModels.jl PRs #1238 (`0fee29a1f8c44a5d3775b7f60f258181ee2580ad`) and #1237 (`503b3797acdf44dba38af1096ebce7fc338f6e9b`) have one-file test changes with passing exact-head Julia shards and documentation previews. Both are five commits behind current main, have no formal reviews, and their paths conflict with other refs in a 33-lane Julia workspace. They remain with their owners. PR #1233 also has passing exact-head CI, but its VA-versus-VA likelihood-ratio contract is unresolved and its test-count claim does not match the added assertions.
- drmTMB and DRModels.jl search returned 35 and 30 open PRs, respectively. Lane preflight reported 31 active R lanes and 33 active Julia lanes. The Julia handoff scan was stopped after it reported 196 unpushed commits across other branches; those branches were not inspected or changed. The owner-attributed worktree inventory remains incomplete.
- The separate CJSModels repositories are present and clean at R `ca7adb6dd3aaee456283b844ab22f4064a14a290` and Julia `80b74c8e7a57eea6fd38c23d59fb90a448ada0d3`. drmTMB PR #1612 remains a 182-file draft at `0b0f4aa7d73835cc4acfd0ca48234f71c2453f72`; keep that CJS integration out of DRM-core cleanup pending its separate-package disposition. Its evidence does not establish interval calibration.
- drmTMB #1600 and DRModels.jl #1167 are already merged at `88e4c7c1b8812e1bec10e191b1b81b89d9bd6bfa` and `6b39b8cf20efba09987a6fa68667160a04ba8f00`, respectively.

The crosswalk candidate remains on `codex/drm-parity-crosswalk-20261102`, separate from #1245. This branch now contains the capability edit and this dated status refresh; no PR was merged or modified during the refresh.

## 11. Team Learning

Memory receipt: relevant parity decision D-203 and the project route were checked. The route script produced no `LOAD-FIRST` manifest. Repository instructions and four page diffs were reviewed; other refs and worktrees were left untouched.

Golden Set: not applicable to a documentation-only reconciliation.

## 12. Cross-Product Coverage

Covers: current-main Julia capability-status wording, two narrowly scoped native routes, source/test citations, and documented parity limits.

Does NOT cover: model tests, the full Julia suite, rendered-site visual review, R–Julia parity, interval calibration, PR creation/updates, merges, releases, or publication.
