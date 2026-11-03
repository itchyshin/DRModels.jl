# After-task: clarify two Julia route boundaries

## 1. Goal

Update the capability ledger with the source-supported limits for ordinary Laplace and coupled NB2 location-scale routes, after reviewing competing page edits.

## 2. Implemented

Clarified that explicit `marginal = :Laplace` differs from default `:LA`, and that its independent per-group objective test does not establish general R–Julia parity or interval calibration. Clarified that coupled NB2 requires matching tag, group, and structure in both formulas and uses iid groups (`Q = I`).

## 3a. Decisions and Rejected Alternatives

- Changed only the two route descriptions; kept the existing 51-row capability snapshot and all status labels.
- Did not add the dirty worktree's Beta known-matrix candidate to current main. It is a separate capability and remains unmerged.
- Preserved the current page edits in the dirty covariance worktree and all four competing refs. The prior user authorization to take this page applies after the fresh diff review.
- Assumed the citation checker is the relevant static gate for this internal design page. If a broader docs build is required, run it on a dedicated exact-source checkout.

## 4. Files Touched

- `docs/design/capability-status.md` (modified)
- `docs/dev-log/check-log.d/2026-11-03-capability-status-route-boundaries.md` (created)
- `docs/dev-log/after-task/2026-11-03-capability-status-route-boundaries.md` (created)

## 5. Checks Run

- `python3 tools/check_capability_citations.py`: passed; 2 citations verified against source, 22 skipped as not precisely resolvable.
- `git diff --check`: passed.
- The source and test evidence was inspected for both documented routes. No package tests or model fits were run.

## 6. Tests of the Tests

Not applicable. This slice changes documentation only. The citation checker verifies resolvable source links; its 22 skipped citations remain outside that guarantee.

## 7a. Issue Ledger

- No issue closed or opened.
- The page has four competing refs and a dirty covariance candidate checkout. This change is limited to the two reviewed paragraphs; owner disposition for the other versions remains open.

## 8. Consistency Audit

Compared the current-main paragraphs with `src/ordinary_laplace.jl`, `test/test_ordinary_laplace.jl`, `src/locscale_frontend.jl`, and `test/test_locscale_frontend.jl`. The opt-in route and its independent objective test support the stated approximation boundary. The parser requires both formulas to carry an identical coupled tag/group/structure; the public test exercises the iid route. No broader parity or calibration claim was added. The separate dirty Beta known-matrix candidate was not copied into this page.

## 9. What Did Not Go Smoothly

The lane census found 28 live lanes and four refs with changes to this page. Diff review showed that those refs contain broader or older snapshots; the patch therefore stays within two current-main paragraphs.

## 10. Known Residuals

- The citation checker skipped 22 citations it could not resolve precisely.
- No package suite, full documentation build, interval calibration, or cross-package parity campaign was run.
- No merge, PR update, or publication action was taken.

## 11. Team Learning

Memory receipt: `route.py` found no LOAD-FIRST manifest in this worktree. Graft supplied the code locations before source reads. Golden Set: not in scope. Durable lesson: record estimator-default differences beside the capability row so implementation presence is not mistaken for estimator parity.

## 12. Cross-Product Coverage

Covers: documentation of the Julia ordinary `:Laplace` and coupled NB2 location-scale routes, their tested scope, and explicit boundaries.

Does NOT cover: R source changes, R-to-Julia bridge admission, other family or covariance routes, repeated recovery, interval availability or calibration, or package-wide parity.
