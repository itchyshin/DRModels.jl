## 1. Goal

Add an optional Gamma density on the natural SD of one named iid Gaussian mean random intercept, retaining existing fixed-coefficient priors and data-likelihood semantics.

## 2. Implemented

Added a shared prior-specification type, an SD Gamma constructor and target preview, exact target and parameter validation, a normalized once-per-SD objective term, separate data-likelihood and penalty metadata, mixed-prior replay, convergence checks, and existing active-prior inference refusals. A standalone example shows a mixed-prior fit.

## 3a. Decisions and Rejected Alternatives

The density is with respect to the natural SD, with no log-SD Jacobian. The target is one unique unlabelled iid mean intercept; broader structures are refused. Assumptions: none beyond the task specification. No default prior was introduced. The historical fixed-only collection and inferential messages are retained.

## 4. Files Touched

Native implementation: `src/DRModels.jl`, `src/fixed_priors.jl`, `src/gaussian_core.jl`, `src/gaussian_ranef.jl`, `src/inference.jl`, `src/introspection.jl`, `src/visualization.jl`, `src/heritability.jl`, and `src/bias_correct.jl`. Validation and reader materials: `test/test_ranef_sd_gamma_priors.jl`, `docs/src/priors.md`, `docs/examples/ranef_sd_gamma_prior.jl`, this report, `docs/dev-log/check-log.d/2026-11-09-ranef-sd-gamma-prior.md`, and `news/ranef-sd-gamma-prior.md`. The task-local `.unlazy/ranef-sd-gamma-julia.md` and external evidence receipts are not package artifacts. Concurrent `src/bridge.jl` and bridge-test changes belong to another lane and are not part of this report.

## 5. Checks Run

At base commit `dd0d984b84144ef19580d15c3392beeba841062d`, the focused native test passed 203 assertions, existing fixed-prior files passed 153, and `test/test_phylo_penalty.jl` passed 73. The focused test compared objective, gradient and Hessian at 30 frozen independent oracle case/point combinations across two shape/rate pairs. The standalone example exited 0 and reported the resolved target, data likelihood and two penalties. Commands, log paths, exact source hashes, concurrent working-tree snapshots, timings, and warnings are in the task evidence receipt `JULIA-FINAL-RERUN.md`; the example log is `julia-native-worked-example.log` in the same task evidence directory. The checks used Julia threads capped at four and BLAS at one. `git diff --check` passed on the combined tree before these closeout files were added. The after-task structure validator passed when run from this checkout. The hub closeout compiler returned failure because it also scanned unrelated hub acceptance ledgers with unmet gates; it did not validate this slice's ledger to completion.

## 6. Tests of the Tests

The initial test failed because the constructor was absent. Subsequent review tests failed for a missing common prior type, a forged selector that preview accepted, and a misleading missing-predictor row-drop warning; they passed after the fixes. The independently reviewed oracle distinguishes omitted, doubled and per-group penalties, shape/rate reversal, a Jacobian, an omitted normalizer and an incorrect target. Three reference fits whose optimizers reported precision loss were compared by objective and finite score rather than parameters.

## 7a. Issue Ledger

No issue number was assigned to this isolated native slice. Upstream PRs overlap Gaussian implementation files and require reconciliation before integration. The bridge belongs to a separate lane.

## 8. Consistency Audit

Checked preview versus fit target and missing-predictor handling, fixed and SD penalty composition, no-prior and fixed-only paths, data log likelihood, update/replay, inference guards, and existing phylogenetic-penalty tests. The SD-only route has no imposed optimizer floor; a separate numerical-interior check rejects underflow even if the score is small. Mixed prior order is retained for replay.

## 9. What Did Not Go Smoothly

The initial clone needed a resolved Julia environment and a task-owned depot overlay. A temporary error-message regression recursively called its own label helper; fixed-prior tests caught it, and the corrected rerun passed. A missing-response conversion error was replaced by early SD-prior validation. Review identified further early-predictor and forged-struct validation gaps, both covered by failing and passing checks. A first closeout template command resolved a relative target inside the hub; that newly created file was removed immediately and the template recreated at the intended absolute path.

## 10. Known Residuals

The full package suite was not run. This report does not establish bridge acceptance, optimizer global optimality, SD recovery, interval calibration, coverage, broad family support, release readiness, or parity on other model structures. The frozen fixture used three groups; its true-zero case is not an SD-recovery claim. The task-local acceptance ledger's manual independent-review gate remains open and its corrected command bindings have not been reverified by the ledger checker.

## 11. Team Learning

Memory receipt: the canonical operating contract, coordination-board check, overlap review, exact path lease, and no-GPL-source boundary shaped the work. The routing tool returned no LOAD-FIRST manifest for this clone or the short repository name; the repository and frozen oracle remained the technical sources. Golden Set: not run; no memory-regression claim is made. A nearly flat Gamma density can give a small score at numerical underflow, so a score check alone does not establish an interior result.

## 12. Cross-Product Coverage

Covers: native Gaussian ML with complete data, unit weights, one iid mean random intercept, natural-SD Gamma prior and optional fixed normal/ridge priors; preview, fitting, separate likelihood/penalty reporting, replay and active-prior guards were exercised. The dedicated existing phylogenetic-penalty test passed.

Does NOT cover: other families, multiple or structured random-effect blocks, slopes, SD submodels, REML, non-unit weights, missing data, broad uncertainty validity, the bridge integration, the full suite, merges or releases.
