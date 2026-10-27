## 1. Goal

Repair current Julia documentation, API help, executable examples, and evidence limits.

## 2. Implemented

Correct displayed formula constructors and keyword calls, describe route-specific integration defaults, qualify capability and cross-family claims, and separate historical benchmark records. Nine diagnostic literals now describe the existing default integration route accurately.

## 3a. Decisions and Rejected Alternatives

Preserve existing numerical algorithms, signatures, and admission boundaries. State the limits of recorded numerical evidence rather than add scientific claims. Keep historical benchmark reports unchanged.

## 4. Files Touched

- `README.md`
- `docs/dev-log/after-task/2026-10-25-docs-audit-current.md`
- `docs/src/api-stability.md`
- `docs/src/capabilities.md`
- `docs/src/cross-family.md`
- `docs/src/diagnostics-and-validation/exact-gaussian-diagnostics.md`
- `docs/src/diagnostics-and-validation/small-sample-behaviour.md`
- `docs/src/drmtmb-parity.md`
- `docs/src/families.md`
- `docs/src/getting-started.md`
- `docs/src/model-guides/convergence.md`
- `docs/src/model-guides/cross-family-methods.md`
- `docs/src/model-guides/large-data.md`
- `docs/src/model-guides/marginal-la-vs-va.md`
- `docs/src/model-guides/model-map.md`
- `docs/src/model-guides/model-workflow.md`
- `docs/src/model-guides/which-scale.md`
- `docs/src/reference/model-fitting-and-postfit.md`
- `docs/src/reference/model-specification.md`
- `docs/src/rosetta.md`
- `docs/src/tutorials/meta-analysis.md`
- `docs/src/tutorials/robust-student.md`
- `docs/src/tutorials/spatial-models.md`
- `docs/src/tutorials/temporal-random-effects.md`
- `news/docs-audit-examples.md`
- `src/DRModels.jl`
- `src/associate_pairs.jl`
- `src/beta.jl`
- `src/betabinomial.jl`
- `src/binomial.jl`
- `src/bivariate_lognormal.jl`
- `src/bivariate_student.jl`
- `src/comparison.jl`
- `src/control.jl`
- `src/cumulative.jl`
- `src/gamma.jl`
- `src/gaussian_core.jl`
- `src/gaussian_lss.jl`
- `src/gaussian_structured.jl`
- `src/lognormal.jl`
- `src/missing_data.jl`
- `src/negbinomial.jl`
- `src/phylo_penalty.jl`
- `src/poisson.jl`
- `src/skewnormal.jl`
- `src/student.jl`
- `src/truncated_poisson.jl`
- `src/tweedie.jl`
- `src/variational.jl`
- `src/zeroonebeta.jl`
- `test/test_docstring_formula_examples.jl`
- `test/test_ordinary_laplace.jl`
- `tools/tests/test_docs_audit_semantics.py`

## 5. Checks Run

The focused capabilities PR passed its strict documentation build and source checks. Current DocString constructor and diagnostic regressions passed 93 assertions. Public manual, DocString, bootstrap, and bridge examples have raw receipts. README model and benchmark examples passed; the displayed output table matches. The current strict build also passed. Independent review aligned the parity benchmark qualification with the README. The reference also describes control reports, loglikelihood, prepared-model overload defaults, and finite-state return semantics; these final additions passed the strict v12 rebuild.

## 6. Tests of the Tests

The task verification harness passed 19 disposable controls, including missing tutorials, soft 404 responses, malformed tables, missing example logs, stale API defaults, incorrect diagnostics, protected-path changes, and build success without deployment. Julia AST controls reject changed executable code and unrelated string edits. R builder tests reject the malformed script while preserving adjacent scripts and idempotence.

## 7a. Issue Ledger

Corrected source findings are part of this patch. Final inventory reconciliation, independent review, required checks, deployment, and direct live checks remain open. The complete task ledger records these as unmet.

## 8. Consistency Audit

Reviewed adjacent reference pages, manual navigation, constructor examples, family restrictions, parameter scales, named interval targets, and retained evidence. Generated pages are checked separately from source pages and aliases.

## 9. What Did Not Go Smoothly

Sparse source exports initially omitted needed assets or extensions. Those exports were corrected and affected checks rerun. Some task runners initially used the wrong fixture or installed package; corrected runs bind the loaded source and preserve earlier failures. Environment failures do not count as successful execution.

## 10. Known Residuals

This report describes a prepared patch, not a completed site audit or publication receipt. Final deployment and full acceptance reconciliation are still required. No package release, scientific calibration, or new capability admission is claimed.

## 11. Team Learning

Memory receipt: exact-source deployment and scoped ownership guidance applied. The task retains raw logs, source hashes, negative controls, and independent review reports outside the package.

Golden Set: the catalog and detector self-tests were checked; the self-tests discriminate their saved good and bad cases. This does not certify the audit transcript or package methods. The task-specific negative controls are recorded separately above.

## 12. Cross-Product Coverage

Covers: current public documentation and the bounded corrections listed above. The paired package has a separate source patch and execution ledger.

Does NOT cover: historical documentation rewrites, releases, tags, new methods, parity or coverage campaigns. Julia docs/design/capability-status.md is excluded and unchanged by this lane.
