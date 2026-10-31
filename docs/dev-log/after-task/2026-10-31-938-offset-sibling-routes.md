# #938: sibling-route guard for `offset()`

## 1. Goal

Re-check #938 (`offset()` estimated as a free regressor for NegBinomial2) on
main after #1211, and check the sibling code paths for the same silent wrong
model: an offset that gets a free coefficient, or one that is stripped from the
formula and never applied.

## 2. Implemented

- No source change. The original Owls repro on main 64ccb2d8d gives NB2
  `coef mu = 0.6526, -0.5200`, logLik -1748.9605, matching drmTMB (-1748.961).
- `test/test_offset_no_silent_route_938.jl` pins the invariant on routes
  `test_offset_fixed_coef_938.jl` does not cover: 15 families x five
  random-effect routes (`(1|g)` default, `:Laplace`, `:VA`, `(1+x|g)`,
  `(1|g)+(1|h)`), `zi`/`hu` for NB2 and Poisson, and bivariate Gaussian,
  Student, LogNormal and mixed-family fits. Each route must throw, or fit with no
  `offset(...)` coefficient, an intercept (or cutpoints) moved by exactly -c for
  a constant offset c, and the log-likelihood unchanged. The routes #1211
  documents as applying the offset must keep applying it.

## 3a. Decisions and Rejected Alternatives

- No new offset support on refused routes: that is a feature decision, not part
  of the bug. The test accepts a refusal on those routes.

## 4. Files Touched

- `test/test_offset_no_silent_route_938.jl` (new)
- `docs/dev-log/check-log.d/2026-10-31-938-offset-sibling-routes.md`
- this note

## 5. Checks Run

See the check-log row. New file 96/96. Full suite: every file passes except
three that fail the same way standalone on main and do not touch offsets.

## 6. Tests of the Tests

- `_add_offset`/`_eta_with_offset` mutated to drop the offset: 8 failures.
- `_design` backstop disabled: the 5 bivariate/mixed-family refusals fail.

## 7a. Issue Ledger

- #938: the reported bug is fixed on main by #1211. Still open on the issue:
  the drmTMB-side bridge refusal (`R/julia-bridge.R`, other repo) and offsets on
  routes that now refuse. Whether to close #938 is left to the owner.

## 8. Consistency Audit

Every univariate and bivariate route either applies the offset at coefficient 1
or raises an error; none estimates or drops it.

## 9. What Did Not Go Smoothly

A failing file aborts its shard, so shards 2 and 3 had to be finished with a
continue-on-failure driver.

## 10. Known Residuals

Pre-existing failures on main, not filed here: `test_bridge_bootstrap_tree.jl`
(non-finite SEs), `test_reml_q4_missing_response.jl` (REML not converged),
`test_temporal_ou.jl` (1.53e-10 vs a 1e-10 tolerance). Julia 1.10 and CI not run.

## 11. Team Learning

Silent-drop is the twin of silent-free: a constant-offset shift test catches both.
