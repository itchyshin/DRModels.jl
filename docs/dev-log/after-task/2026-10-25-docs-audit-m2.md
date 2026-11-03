# Capability-page repair checkpoint

## 1. Goal

Repair the Julia public capability page's broken table qualification and its inconsistent distinction between numerical tests and complete worked analyses. This is the focused Julia milestone within the two-package documentation audit.

## 2. Implemented

Kept the full R-bridge qualification in its status cell, repaired the neighboring deprecated-alias row, and defined Tested as bounded numerical evidence. Beta and Gamma crossed-effect rows now say that complete worked analyses are absent. Preserved the prior owner's single sleepstudy comparison and its evidence limits. Retained newer main corrections while rebuilding the patch on source 19f7edb3. Clarified the Beta supplied-matrix convention and the remaining scale-axis interval-coverage limit.

## 3a. Decisions and Rejected Alternatives

Kept the existing numerical status instead of inventing analyses or changing capability admission. Rejected copying an older complete page because that would lose newer main corrections. Assumptions made without asking: this patch concerns presentation and evidence scope; it changes no model, estimand, algorithm, release, or protected design ledger.

## 4. Files Touched

- `.github/workflows/Documenter.yml`
- `docs/src/capabilities.md`
- `tools/tests/test_docs_audit_semantics.py`
- `docs/dev-log/check-log.d/2026-10-25-docs-audit.md`
- `docs/dev-log/after-task/2026-10-25-docs-audit-m2.md`
- `news/docs-audit-capabilities.md`

Other audit changes remain separately owned and uncommitted; they are outside this focused patch.

## 5. Checks Run

Six focused source/table regression tests passed against the committed capability page. The first PR check also ran five broader audit tests whose repairs were not in this slice; those tests were separated into the broader audit before rerunning the focused check. The existing reader-check CI step now runs them; deployment triggers and protections are unchanged. A strict Documenter build on the 19f7 audit snapshot passed. Its 48 generated HTML pages had zero unexpected local link, asset, fragment, or visible raw-math failures; the deployment-root versions.js reference is separately recorded. Rendered capability assertions passed after correcting the checker's inline-text normalization. The final pointer wording still requires exact-head CI and publication verification before this milestone can be called complete.

## 6. Tests of the Tests

Disposable malformed rows and lost qualifications were rejected by the source checker. Additional mutations detect stale ZeroOneBeta, LogNormal, phylogenetic-scale, and bootstrap-generator wording. No production page was changed to manufacture a failing control.

## 7a. Issue Ledger

JTABLE and JSTATUS have source and local-render evidence. Exact-head CI, main deployment, and direct live browser verification remain required. The R tutorial restoration has separate source, deployment, artifact, and live receipts. Broader page, API, and example findings remain in the audit ledger.

## 8. Consistency Audit

Checked adjacent table rows, the REML restriction paragraph, the bootstrap pointer, newer-main rows, and Beta matrix semantics. Confirmed that this patch changes neither `docs/design/capability-status.md` nor numerical source. The owner's public-page release and successful scoped ownership claims are recorded in the audit task.

## 9. What Did Not Go Smoothly

An early full-page proposal was based on an older snapshot; it was rejected before landing. One independent review initially repeated stale row claims and corrected its receipt after rereading current bytes. The rendered checker's whitespace handling was corrected and rerun. These events are retained rather than hidden by a final pass label.

## 10. Known Residuals

Publication is not established by a local build. Required exact-head checks and the deployment job must succeed, followed by direct live verification. The full documentation and executable-example audit remains in progress. No interval-calibration, route-wide recovery, or package-release claim is made.

## 11. Team Learning

Preserve a prior owner's focused correction, then apply it to current main rather than copying their whole older file. Check qualifications in rendered cells as well as source. Memory receipt: routed ownership, source-provenance, and deployment-evidence guidance shaped this slice. The Golden Set catalog and detector self-tests were checked; they are heuristic guard checks, not scientific validation or a transcript-level completion verdict.

## 12. Cross-Product Coverage

This slice covers the public Julia capability table and its evidence qualifications. It does NOT cover the protected design ledger, new methods or admissions, numerical calibration, stable-version retagging, package releases, or completion of the broader R/Julia documentation audit. The Beta matrix note describes existing route differences; it does NOT cover a claim of general cross-package parity.
