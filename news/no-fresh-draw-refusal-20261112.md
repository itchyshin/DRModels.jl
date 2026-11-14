- **`bootstrap_summary`, `bootstrap_result`, `bootstrap_ci` and the default `simulate` now refuse Gamma and Binomial fits with two or more ordinary random-effect terms.**
  This is an explicit change: a call that used to return a table (or fail late) now stops with an error.
  The refused shapes are `(1 | g) + (0 + x | g)`,
  mixed shapes such as `(1 | g) + (0 + x | g) + (1 | h)`,
  and slope-only shapes such as `(0 + x | g) + (0 + z | g)`.
  None of them has a fresh random-effect draw. Crossed or nested intercepts such as `(1 | g) + (1 | h)` do have one
  (the simulator for several random intercepts) and are not refused. Until now the bootstrap quietly drew responses with the
  random effects set to zero, so the variance-component interval collapses.
  The message names the terms you fitted and points to `simulate(fit; data = data, re = :zero)`
  for a conditional draw where the fitted object supports it. Binomial and the other families share one
  message form and one rule (see the separate notes on Binomial and on slopes).
  A single `(1 | g)` is unchanged for both families. Gamma `(0 + x | g)` and `(1 + x | g)` are refused too (see the separate note on Gamma slopes).
  This is a maintainer decision: refuse until the fresh draw exists.
