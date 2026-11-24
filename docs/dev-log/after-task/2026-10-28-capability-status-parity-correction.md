## 1. Goal

Reconcile Julia's internal capability page with current-main source and tests after reviewing the three divergent page versions reported by lane preflight.

## 2. Implemented

Added two scoped rows for the non-Gaussian ordinary `marginal = :Laplace` intercept route and the coupled NB2 ordinary location-scale intercept. Clarified that the Julia bridge status describes an available public surface, not parity across all routes. Recounted the page's capability table and corrected its snapshot from 46 to 51 rows. Follow-up diff review also corrected the cross-family bivariate status from `missing` to `experimental`, matching current-main code and tests plus the recorded non-promotion decision.

## 3a. Decisions and Rejected Alternatives

- Reviewed all three differing refs before editing. The `rename/drmodels-20260917` ref is older than current main and omits the current bivariate refusal detail. `codex/071-ordinary-laplace` contains the two useful route rows; I retained them with narrower wording checked against current source and tests. `claude/capability-status-parity` is an earlier pre-rename page with stale content, so I did not copy its snapshot or prose wholesale.
- Reviewed the later overlapping diffs from `capability-status-reconcile`, `cross-family-interval-language`, `parity-provenance-docs`, and `known-covariance-parity`. Their cross-family evidence agreed with the exact-main A11 decision; their unrelated citations and candidate-only covariance wording were left on their source refs.
- Kept the ordinary Laplace claim to Poisson, Binomial, NegBinomial2, Gamma, and Beta with one unstructured mean intercept, fixed `sigma ~ 1` for scale families, and ML.
- Kept the coupled location-scale claim to the publicly tested NB2 iid route. I did not extend it to Gamma or structured groups based on adjacent source alone.
- Kept bridge status separate from route-wide parity and statistical interval coverage.
- Kept the cross-family row `experimental`: current main includes the formula route and tests, while the recorded A11 decision explicitly says not to promote it without a later decision.

## 4. Files Touched

- Updated `docs/design/capability-status.md`.
- Added this report and `docs/dev-log/check-log.d/2026-10-28-capability-status-parity-correction.md`.

## 5. Checks Run

### 2026-11-01 source-pin and citation recheck

- Confirmed the candidate page still names the live drmTMB main `b535bfbcce2a9039d869c2e34fe9da476394bf62` and DRModels.jl main `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`.
- Ran `python3 tools/check_capability_citations.py` against a clean archive of DRModels.jl `origin/main` with this candidate page copied in: 2 citations verified; 18 citations were skipped because the checker cannot resolve them precisely.
- `git diff --check -- docs/design/capability-status.md`: passed.
- Rechecked this report with `Rscript ~/shinichi-brain/tools/check-after-task.R` and the naturalness checker: structure passed; 1 style flag in 1,184 words, 0 findings.
- No Julia model tests or documentation build were rerun in this recheck. The focused source-and-test results below are from the same exact Julia source commit; no Julia main change was observed.

### Original 2026-10-28 focused checks

- Inspected exact current-main source and tests: `src/ordinary_laplace.jl`, its include in `src/DRModels.jl`, the auto-discovered `test/test_ordinary_laplace.jl`, `src/negbinomial.jl`, `src/locscale_frontend.jl`, and `test/test_locscale_frontend.jl`.
- Recounted the status table by status-cell parser: 51 rows, 45 implemented, 2 rejected, 1 planned, 2 experimental, and 1 missing. These totals match the revised snapshot.
- Confirmed in exact current-main source that `src/mixed_family.jl`, `src/mixed_family_postfit.jl`, and the three named test files exist. The retained A11 check log reports the formula test passed 18 assertions and explicitly keeps the route experimental. The current public capability guide also labels the route experimental. No cross-family test was rerun in this follow-up.
- Queried GitHub and confirmed R `main` is `b535bfbcce2a9039d869c2e34fe9da476394bf62`; read that exact source's `inst/extdata/julia-capabilities.tsv`: 37 routes, bridge statuses 17 supported, 6 partial, 13 experimental, 1 unsupported; claim statuses 10 covered, 22 partial, 4 experimental, 1 unsupported. The capability page now names this pin so its cross-repository counts are reproducible.
- Ran `test/test_ordinary_laplace.jl` in an isolated temporary Julia environment against source commit `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`: 335/335 passed in 1m47.9s. Test SHA-256 `30f357339d616de56ebd174b797393fa1ed89ae4cb38dd2309f7665cb822c73d`; output SHA-256 `e136b7b6b2ffa2e625ce04be78c69307591298bd5121318f720e6605f7a3e1a1`.
- Ran `test/test_locscale_frontend.jl` in the same temporary environment: location-scale routing 10/10 and lone coupled-tag refusal 1/1 passed. Test SHA-256 `e4f21caa37215c2d89fa4199f9b55817bf8122b61169700ed845960a0ef97b7c`; output SHA-256 `afc99482b7d3e910c1a4b0b43b3fc7c7a12dbf714d514d58c333906cb62baa04`.
- Temporary environment used Julia 1.10.0, `Distributions` 0.25.132, and `ForwardDiff` 1.4.6, with one Julia and one BLAS thread. The environment and depot are outside the repository; no dependency or manifest file in the checkout changed.
- `git diff --check` passed. The naturalness check found 0 issues.
- The first direct run could not load dependencies because the checkout has no manifest; after creating the isolated test environment, both focused files passed. No full Julia suite or documentation build was run; this remains an internal ledger update, not a public release page.

## 6. Tests of the Tests

No test was added or changed. The two existing default-suite files named by the capability rows passed in isolation. This does not establish the full suite or route-wide parity.

## 7a. Issue Ledger

- Verified from current source: the ordinary Laplace public route admits the stated five families and scoped intercept shape.
- Verified from current source and an existing public test: coupled NB2 location-scale iid fitting is routed through the public formula frontend.
- Corrected: the previous capability snapshot undercounted the current table by three rows; the updated table has 51 rows. The cross-family row is now `experimental`, not `missing`, and the status counts reflect that change.
- Still open: current full Julia CI has failing Julia 1.10 shards; bridge routes remain partial by route; docs preview owners still need to reconcile the public parity page. Focused tests do not establish the full suite, route-wide parity, or interval validity.
- Deferred: CMR Phase B interval availability and coverage/calibration.

## 8. Consistency Audit

The two new `implemented` labels use the page's stated source-plus-test standard. Neither label claims full R parity, calibrated uncertainty, or a universal random-effect route. The cross-family `experimental` label now matches the recorded non-promotion decision and the public capability guide. The bridge row points readers to the route-specific R ledger and distinguishes its `claim_status` labels from statistical interval coverage.

## 9. What Did Not Go Smoothly

The alternate refs were not clean successive versions: they diverge from current main, include old package names, and mix current-main facts with candidate-only wording. Reviewing their actual diffs avoided importing an older page wholesale. The original page's `missing` cross-family label also contradicted its retained A11 decision and the public capability guide.

## 10. Known Residuals

The capability page remains an internal status view rather than a completed route-by-route parity audit. This update does not repair the current Julia full-suite failures, change public documentation, or validate standard errors and intervals for the two new routes. Cross-family current-source execution was not rerun in this follow-up; its status is `experimental` based on the existing source, test, and decision record. The R-side route counts describe the live `main` pin above and will need refreshing when that source advances.

## 11. Team Learning

When several historical refs modify a status page, compare each with current main and validate each retained claim against current source, tests, and decision records. A snapshot count should be computed from the table instead of carried forward.

Memory receipt: The DRM parity note in `MEMORY.md` supports exact-source accounting and keeping focused test evidence separate from package-level claims. `route.py` returned no LOAD-FIRST manifest for this worktree, and the `graft` command was unavailable.

Golden Set: Not run. This was a bounded documentation audit, not preference-learning work.

## 12. Cross-Product Coverage

Covers: current Julia capability wording for ordinary non-Gaussian Laplace, coupled NB2 iid location-scale, cross-family bivariate status, and the scope of the R-to-Julia bridge status.

Does NOT cover: full package tests, route-wide R/Julia parity, standard-error or interval validation, calibration, release readiness, or CMR Phase B.
