## 1. Goal

Add an optional Gamma density on the natural SD of one named iid Gaussian mean random intercept, retaining existing fixed-coefficient priors and data-likelihood semantics.

## 2. Implemented

Added a shared prior-specification type, an SD Gamma constructor and target preview, exact target and parameter validation, a normalized once-per-SD objective term, separate data-likelihood and penalty metadata, mixed-prior replay, convergence checks, and existing active-prior inference refusals. A standalone example shows a mixed-prior fit.

## 3a. Decisions and Rejected Alternatives

The density is with respect to the natural SD, with no log-SD Jacobian. The target is one unique unlabelled iid mean intercept; broader visible structures are refused. StatsModels collapses repeated identical written terms before target resolution, as described under Known Residuals. No default prior was introduced. The historical fixed-only collection and inferential messages are retained.

## 4. Files Touched

Native implementation: `src/DRModels.jl`, `src/fixed_priors.jl`, `src/gaussian_core.jl`, `src/gaussian_ranef.jl`, `src/inference.jl`, `src/introspection.jl`, `src/visualization.jl`, `src/heritability.jl`, and `src/bias_correct.jl`. The integrated candidate also changes `src/bridge.jl` and bridge tests. Validation and reader materials include `test/test_ranef_sd_gamma_priors.jl`, `docs/src/priors.md`, `docs/examples/ranef_sd_gamma_prior.jl`, this report, `docs/dev-log/check-log.d/2026-11-09-ranef-sd-gamma-prior.md`, and `news/ranef-sd-gamma-prior.md`. Task-local acceptance ledgers and external evidence receipts are not package artifacts.

## 5. Checks Run

In the isolated checkout based on `dd0d984b84144ef19580d15c3392beeba841062d`, with the native edits applied, the focused native test passed 203 assertions, existing fixed-prior files passed 153, and `test/test_phylo_penalty.jl` passed 73. The focused test compared objective, gradient and Hessian at 30 frozen independent oracle case/point combinations across two shape/rate pairs. The standalone example exited 0 and reported the resolved target, data likelihood and two penalties. Commands, log paths, exact source hashes, concurrent working-tree snapshots, timings, and warnings are in the task evidence receipt `JULIA-FINAL-RERUN.md`; the example log is `julia-native-worked-example.log` in the same task evidence directory. The checks used Julia threads capped at four and BLAS at one. `git diff --check` passed on the combined tree before these closeout files were added. The after-task structure validator passed when run from this checkout. The hub closeout compiler returned failure because it also scanned unrelated hub acceptance ledgers with unmet gates; it did not validate this slice's ledger to completion.

The later candidate `0692e82e8e8b0dd6a930fa04f57cf4611b691f8a` passed 312 focused API, bridge and refusal assertions, and 203 frozen-oracle assertions, on Totoro with Julia 1.13.2. It also passed seven live paired bridge tests with R commit `95c621349b8ed91ae4afec0a07722346b6ed5a62` on Julia 1.10.0. `julia-1132-focused.log`, `julia-1132-oracle.log` and `PAIRED-LIVE-FINAL.md` in the task evidence record the test commands and scope. The paired run recorded matching hashes for selected bridge and test files; it did not establish full-tree provenance.

## 6. Tests of the Tests

The initial test failed because the constructor was absent. Subsequent review tests failed for a missing common prior type, a forged selector that preview accepted, and a misleading missing-predictor row-drop warning; they passed after the fixes. The independently reviewed oracle distinguishes omitted, doubled and per-group penalties, shape/rate reversal, a Jacobian, an omitted normalizer and an incorrect target. Three reference fits whose optimizers reported precision loss were compared by objective and finite score rather than parameters.

## 7a. Issue Ledger

No issue number was assigned to this isolated slice. Draft PR #1281 is open; exact-head CI remains required. Upstream PRs overlap Gaussian implementation files and require reconciliation before integration.

## 8. Consistency Audit

Checked preview versus fit target and missing-predictor handling, fixed and SD penalty composition, no-prior and fixed-only paths, data log likelihood, update/replay, inference guards, and existing phylogenetic-penalty tests. The SD-only route has no imposed optimizer floor; a separate numerical-interior check rejects underflow even if the score is small. Mixed prior order is retained for replay.

## 9. What Did Not Go Smoothly

The initial clone needed a resolved Julia environment and a task-owned depot overlay. A temporary error-message regression recursively called its own label helper; fixed-prior tests caught it, and the corrected rerun passed. A missing-response conversion error was replaced by early SD-prior validation. Review identified further early-predictor and forged-struct validation gaps, both covered by failing and passing checks. A first closeout template command resolved a relative target inside the hub; that newly created file was removed immediately and the template recreated at the intended absolute path.

## 10. Known Residuals

The full package suite remains open. A run expected to exceed three hours requires separate approval; the longest historical CI shard took about 3 hours 47 minutes. StatsModels collapses repeated identical written random terms before `bf(::FormulaTerm)` receives them, so Julia cannot reject that spelling separately from a single term at the current API boundary; R rejects it. The maintainer selected the existing syntax with this documented exception for the first slice. The string bridge has its own pre-fit guard. A pinned fixed-only bridge comparison matched 365 numeric values and other compared fields in four cases, but strict identity failed for compiler-generated `diagnostics.route` strings. This report does not establish optimizer global optimality, SD recovery, interval calibration, coverage, broad family support, release readiness, or parity on other model structures. The frozen fixture used three groups; its true-zero case is not an SD-recovery claim. Final acceptance-ledger re-verification remains open.

## 11. Team Learning

Memory receipt: the canonical operating contract, coordination-board check, overlap review, exact path lease, and no-GPL-source boundary shaped the work. The routing tool returned no LOAD-FIRST manifest for this clone or the short repository name; the repository and frozen oracle remained the technical sources. Golden Set: not run; no memory-regression claim is made. A nearly flat Gamma density can give a small score at numerical underflow, so a score check alone does not establish an interior result.

## 12. Cross-Product Coverage

Covers: native Gaussian ML with complete data, unit weights, one iid mean random intercept, natural-SD Gamma prior and optional fixed normal/ridge priors; preview, fitting, separate likelihood/penalty reporting, replay and active-prior guards were exercised. The dedicated existing phylogenetic-penalty test passed.

Does NOT cover: other families, multiple or structured random-effect blocks, slopes, SD submodels, REML, non-unit weights, missing data, broad uncertainty validity, exhaustive bridge formula coverage, the full suite, merges or releases.
