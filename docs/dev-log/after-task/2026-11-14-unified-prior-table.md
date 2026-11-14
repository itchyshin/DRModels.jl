# Unified prior-table adapter

## Scope

The adapter accepts Tables.jl-compatible inputs for fixed-effect normal and ridge penalties and Gamma densities on one or two named Gaussian mean random-intercept SDs. It translates the table into existing prior specifications. The table Gamma form takes mean and shape, while the constructor takes shape and rate. Random slopes, correlations, structured random effects and other families remain outside this slice.

Blank fixed-effect selectors cover non-intercept coefficients. Blank SD selectors cover supported random-intercept SDs. Overlapping selections and coefficients literally named `all` are rejected. The original table is retained, and update replays the resolved targets.

## Evidence

At code commit `59518deab`, the focused Julia table test passed 77 assertions. The dense-Gaussian oracle test passed 460 assertions over 150 points, including gradient and Hessian checks, reordered targets and seven wrong-implementation controls. The parser, preview and update tests passed separately. The fixed-prior and two-intercept prior regression tests passed; the latter reported covariance Hessian warnings in the saved log.

A live R-to-Julia bridge test passed on the isolated R snapshot and Julia source, covering fixed-only, single-SD and two-SD mixed tables. Source hashes matched before and after. The first bridge run had two frame-shape comparison failures; the later test compared the named targets and their fields. The full Julia package suite on this adapter, independent code review and current-main reconciliation remain open. The R snapshot compatibility receipt passed after this run; full package checks remain open.

## Boundary

This is penalized maximum likelihood and does not provide posterior samples or calibrated uncertainty. The R and Julia table branches are draft work; no merge or release is implied by these focused results.
