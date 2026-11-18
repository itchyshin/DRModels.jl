# After-task: independent-oracle contract for the fixed-effect `meta_V` route (PR #1167)

## 1. Goal

Complete PR #1167's required repository records while preserving its test-only scope. The test checks the fixed-effect Gaussian meta-analytic route `y ~ x + meta_V(v)` with independent numerical oracles.

## 2. Implemented

Prepared the test-only PR candidate on exact current main. PR head `9c370bae65d7d1aece260f8a7ed997d475c616f8` was merged with exact main `e4333387050549d49bf95738543e1e11bc81fa8c`; the merge had no content conflicts. The candidate before this records-only commit is `37ebd93fed84cd21d524eb4b07d4563099d23d29`. No production source, API, news, or interval-calibration claim was added.

To preserve the repository's linear-history rule, the same three-file diff was applied directly to current `main`. Its tree `ce3959a702652cf1fabd01346049ab68dc77c5a4` exactly matches the reviewed merged candidate. Focused and shared-`Main` tests were rerun on that exact tree; raw logs are `/private/tmp/pr1167-linear-candidate.log` (SHA-256 `463960d1855b2194ffd46c4b34bd1c18102d4f7d7854ba36ca9df517dc825784`) and `/private/tmp/pr1167-linear-smoke.log` (SHA-256 `73183dfd7ab37e83eaea125f98b29855922d0bda2f7c41b762b223fc304257a6`).

## 3a. Decisions and Rejected Alternatives

Kept the test-only scope because the test exercises an existing route and no implementation defect was in scope. Did not add a NEWS fragment because this adds no user-facing behavior. Did not pin invalid-input behavior or REML behavior: the PR records that invalid-variance handling belongs to PR #1057 and this route refuses REML. Assumed current `main` is the intended base because the PR targets `main`.

## 4. Files Touched

- `test/test_meta_v_fixed_route.jl` (PR test; narrowed `Distributions` import to avoid shadowing `DRModels.Poisson` in the shared `Main` namespace)
- `docs/dev-log/after-task/2026-10-22-meta-v-fixed-route.md` (added)
- `docs/dev-log/check-log.d/2026-10-22-meta-v-fixed-route.md` (added)

## 5. Checks Run

- Exact-source candidate `37ebd93fed84cd21d524eb4b07d4563099d23d29`: focused test passed, 23/23 assertions (16 independent-oracle + 7 Gaussian-limit).
- Shared-`Main` smoke on that candidate: both the new test (23/23) and existing Poisson display test (4/4) passed; `Poisson()` resolves to `DRModels.Poisson` after the selective import.
- `DRM_TEST_LIST_ONLY=1 julia --project=. test/runtests.jl`: confirmed 402 test files and automatic discovery of `test_meta_v_fixed_route.jl`.
- `git diff --check` passed. The check-log renderer reported 40 malformed historical fragments; confirm below whether this new row is among them.
- Julia dependencies were instantiated in an isolated temporary depot. The package server registry download emitted a DNS warning; cached/local registry data resolved dependencies. No tracked Manifest change was made.
- Full suite and GitHub CI were not run, per orchestrator scope; exact-head CI remains pending. No recovery, interval availability, or interval-calibration claim is made.
- The after-task structure checker is invoked directly against this report; the global evidence-promotion gate is outside this test-only scope.

## 6. Tests of the Tests

The PR description reports that deliberate scratch mutations to the log-likelihood, variance split, fitted values, and covariance symmetry caused their corresponding assertions to fail. Those mutation runs were reported by the PR author and were not rerun in this records-only update. The current focused run confirms the unmodified assertions pass.

## 7a. Issue Ledger

- PR #1167: records completed in this candidate; not merged.
- PR description names #1057 as the owner of invalid-variance behavior. No change made to that contract.

## 8. Consistency Audit

Confirmed the source candidate differs from exact main only by the focused test and the two required records; no `src/` files changed. The test file is auto-discovered by the 402-file runner. The selective import preserves `DRModels.Poisson` in shared `Main`. Check-log entry uses the documented five-column row format. The PR remains test-only.

## 9. What Did Not Go Smoothly

The managed worktree shared read-only Git metadata, so I used an isolated clone under `/private/tmp`. Dependency setup reported a package-server DNS failure but local cached registry data sufficed. A test namespace collision with `Distributions.Poisson` surfaced in shared `Main`; narrowing imports fixed it without touching implementation code.

## 10. Known Residuals

The full Julia suite and GitHub CI have not been verified on this exact candidate, as instructed. No benchmark, statistical recovery study, or interval coverage/calibration validation was performed or claimed. The test's deliberate mutation controls are supported by the PR description, not by a fresh run in this update. The check-log directory contains 40 malformed historical files that need a separate cleanup pass. No push or merge was performed. Fresh CI and normal review are still required before merge.

## 11. Team Learning

For test-only PRs, the report should distinguish newly executed checks from mutation evidence supplied in the PR description. When a standard report generator resolves output paths against another repository, do not write outside the authorized checkout; use the required section template locally and disclose any unavailable validator.

Memory receipt: Read the DRModels.jl `AGENTS.md`, `HANDOVER.md`, `ROADMAP.md`, check-log fragment instructions, coordination board, and the shared after-task protocol. The repository handover's old ownership note was historical; this task was explicitly assigned in the active PR-dispatch lane. Golden Set: not in scope for this test-records update.

## 12. Cross-Product Coverage

Covers: fixed-effect Gaussian `meta_V` point-fit likelihood and extraction behavior checked by this test.

The test does NOT cover: random-effect meta-analysis, invalid-variance input behavior, REML, other families, the full test suite, R/Julia parity, statistical recovery, interval availability, or interval coverage/calibration.
