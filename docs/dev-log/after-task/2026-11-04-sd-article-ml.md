# SD regression article: Julia documentation closeout

## 1. Goal

Give an applied ecologist a runnable Julia counterpart to the ML-first drmTMB article. Explain predictor-dependent latent-effect SD with one supplied-matrix Gamma example, while keeping source admission, point validation, interval calibration and integration status separate.

Original premerge snapshot at c037268c7; the integration update below supersedes its test and asset status. Local example and documentation validation is complete for the scope below. Pull-request checks, remote asset availability, merging and post-merge main checks remain pending at this report's source state.

## 2. Implemented

Added `docs/src/tutorials/predictor-dependent-random-sd.md` and placed it immediately after the existing location-scale-scale tutorial in `docs/make.jl`. The page uses public `bf`, `drm`, `coef`, `predict` and `predict_parameters` calls. It distinguishes the estimated constant Gamma coefficient of variation `sigma`, the latent SD multiplier from `sd(id, relmat)`, and marginal latent SD after applying the supplied matrix diagonal.

Three generated data files and a manifest live under `docs/src/assets/sd-regression/`. The assets were committed as `e8a666769fc436e73c7121a4c49b1d9c92866776`. The page links to those immutable objects because the site build did not carry TSV downloads into its final output. No drmTMB GPL source was copied into this MIT repository; only generated synthetic data, numerical results and independently written Julia documentation were used.

## 3a. Decisions and Rejected Alternatives

Retained the existing documentation menus and Gaussian LSS tutorial. The new page adds the non-Gaussian supplied-K example rather than replacing that learning path.

Used Base-only readers for the supplied unquoted TSV assets, avoiding a new data-reader dependency. The page explicitly reorders K using its label sidecar and the data's first-seen group order. It preserves raw K instead of converting it to a correlation matrix.

Assumptions made without asking: the supplied example files use safe group labels and contain no embedded tabs or quoted fields, as verified in the asset check. A general-purpose importer would be needed if that file contract changed. No numerical tolerance was introduced or relaxed: the point check retained the existing absolute log-likelihood tolerance of `1e-5`.

## 4. Files Touched

- `docs/src/tutorials/predictor-dependent-random-sd.md`
- `docs/make.jl`
- `docs/src/assets/sd-regression/gamma-data.tsv`
- `docs/src/assets/sd-regression/gamma-K.tsv`
- `docs/src/assets/sd-regression/gamma-K-labels.tsv`
- `docs/src/assets/sd-regression/manifest.json`
- `docs/dev-log/check-log.d/2026-11-04-sd-article-ml.md`
- `docs/dev-log/after-task/2026-11-04-sd-article-ml.md`
- `news/sd-article-ml.md`

No Julia engine source, likelihood, formula grammar or inference implementation changed.

## 5. Checks Run

The four public Julia code blocks ran together in a fresh session against engine source `f4fa3aed451636b6ffa442ce2632515513d23b1f`. The manifest records source-tree, script, runtime-log and data hashes. Results were log likelihood `-572.3288655328454`, SD coefficients `[-0.5135359187018407, 0.07043084782335204]`, constant Gamma `sigma = 0.4975113016673914`, and one-unit SD multiplier ratio `1.0729703686286995`.

The independent Laplace reference was `-572.328865533046`, an absolute gap of `2.0066e-10`, below `1e-5`. Direct Julia `fitted(fit)` and `predict(fit, dat)` agreed exactly on this example. Both set random effects to zero; they are not an integrated marginal response mean. The checked R bridge uses a conditional retained structured field for stored-row outputs, a different public output contract.

Asset checks passed for 480 strictly positive response rows, 40 groups, group-constant z, non-unit K diagonal, matching labels and recorded hashes. The true one-unit SD multiplier ratio was `1.4918247`, compared with the estimated `1.0729704`. Implementation agreement does not establish recovery of the generating slope from this single draw.

The final full Documenter/VitePress build passed with `DRM_DOCS_DEPLOY=false`; it did not deploy. Browser inspection passed for layout, mathematical notation, navigation, visible point results and limits. All four distinct immutable download targets in the rendered page matched local Git objects at the asset commit. Remote HTTP checks remain pending branch push.

Evidence files retained by the execution lane are `julia-guide-direct.log`, `julia-docs-final-build.log` and `julia-render-visual-check.json` under the 4 November SD article execution receipt directory. The rendered guide and the four executed code blocks were checked for agreement. Prose, exact-draft assessment and after-task structure are checked as part of this closeout; these checks do not substitute for the scientific or integration gates.

## 6. Tests of the Tests

The example's existing independent Laplace tolerance was retained. Its observed likelihood gap is reported without rounding it into a claim of equality. The same fitted data produce different conditional and zero-random-effect response quantities, so the page names the contract instead of treating a changed prediction as evidence of a changed estimator.

The rendering check exposed a real failure of the download assumption: TSV files existed in intermediate output but not the final site. The correction was checked against all four distinct asset targets. No Julia engine mutation test or new numerical regression was added for this documentation-only change.

## 7a. Issue Ledger

The local download-path defect is fixed with immutable asset URLs; remote availability still needs checking after push. The remaining PR, CI, merge and main checks belong to the active first-delivery integration gate. This report closes no broad capability issue and makes no whole-programme completion claim.

G5 interval calibration and D-342 REML policy remain deferred. The R-side structured stored-prediction repair and numerical gap audit are separate parts of the paired first delivery, not Julia engine changes in this slice.

## 8. Consistency Audit

Compared the new page with the existing Gaussian LSS and known-matrix tutorials, source-adjudicated the Gamma route, checked public selector spelling `sd(id, relmat)`, and retained the distinction between raw-matrix multipliers and marginal SD. Reviewed competing navigation diffs before adding one disjoint entry. Rechecked all four distinct asset URLs after the renderer exposed the TSV omission.

Memory receipt: this slice used the parent-approved ML-first plan, its bounded source-adjudication packet, repository instructions and the prose-style-review checklist. The decisions that shaped the work were preserving the Gaussian tutorial, distinguishing interfaces, retaining raw K and not converting a source or point check into a calibration claim. No new cross-project research or durable memory edits were performed in this documentation slice. The Golden Set was not run; the relevant repeated-mistake check was the explicit sweep of every download target and the displayed scientific quantities.

## 9. What Did Not Go Smoothly

A successful source-page check did not prove that the built site would serve the example data. The renderer omitted the TSV assets from final output, so the download links were changed to immutable generated-data Git objects and checked in the actual HTML.

The fitted SD slope was smaller than the generating slope on this draw. The page retains that discrepancy rather than selecting a more favourable example or claiming parameter recovery.

## 10. Known Residuals

The immutable URLs are locally verified Git objects but have not yet passed remote HTTP checks at this report's state. PR checks, merge verification and post-merge main checks remain pending. The browser check establishes local rendering, not successful public deployment.

This example provides no interval coverage qualification, calibrated uncertainty, REML policy resolution, broader family/provider validation or full Julia test-suite result. The Julia source is unchanged; the focused public example and full documentation build are the checks relevant to this slice.

## 11. Team Learning

Validate downloadable data in the final rendered site. Source presence and intermediate build output are insufficient. Immutable generated-data links can bind a public example to inspected bytes, but still require a remote read after push.

Keep interface-specific prediction labels beside the example. Direct Julia zero-effect predictions and the R bridge's conditional stored-row predictions can both be correct while answering different questions.

## 12. Cross-Product Coverage

This slice covers one ML Gamma model with a log mean, one scalar relmat mean intercept, a supplied raw K, a group-constant fixed-predictor SD submodel and one estimated constant response CV. It covers the displayed direct Julia public calls, their point quantities, generated assets, navigation and local rendered page. The paired R numerical comparison is confined to the same generated fixture.

It does NOT cover other families, providers, precision-input matrices, missing responses, random slopes, correlated or crossed fields, residual-scale SD regression, conditional prediction for new groups, profile/bootstrap workflows, uncertainty coverage, G5 calibration, D-342 REML policy, release readiness or whole-programme completion. Direct Julia `fitted` and `predict` on this example use RE=0; the R bridge's checked stored structured outputs condition on its retained field. Their output contracts remain separate.

## Integration update: isolate quadrature-test imports

The latest main CI run 41859055610 was already failing before this article PR. Its Julia 1.10 shard 1 reported an invalid redefinition of Main.D at test_iid_sd_quadrature_accuracy.jl:23: another test uses D for DRModels, while this file used D for Distributions. The repair removes that alias and qualifies its 29 module references explicitly. The remaining three global constant names are unique in the test tree. No engine source, seed, formula, likelihood calculation, assertion or numerical tolerance changes. The header now describes the existing 1e-4 fitted/nearby and 1e-2 far-stress thresholds accurately.

An isolated Julia 1.10.12 environment reproduces the exact CI error before repair. With the repaired file, the namespace smoke passes three assertions on both Julia 1.10.12 and 1.13.0. It uses the existing DRM_QUAD_EXPLORE guard to load helpers without executing the fit study; these passes do not qualify the full quadrature grid. A separate three-assertion AST comparison verifies equivalence after module qualification and detects a deliberately changed tolerance. Both environments are external to the checkout; its engine source and package project files are unchanged. Setup failures from a Julia 1.13 manifest under 1.10 and initially missing test-only dependencies remain in the external logs.

The current PR's stable-Julia shard 3 passed the Binomial fixture that failed in the earlier main run. That single pass does not diagnose the historical nonconvergence. Another active lane owns test_binomial_sd_public.jl; this repair does not alter it. Earlier main-run Gaussian-warning and temporal-oracle failures, and measured Tweedie test latency, are retained as historical diagnostics rather than counted as current reproduced product defects.

Source, scripts, before/after checks and targeted CI logs are retained under /Users/z3437171/Documents/Codex/2026-11-04-sd-article-plan/execution/. Full new-head PR checks and post-merge main checks remain pending. G5 and D-342 remain deferred.
