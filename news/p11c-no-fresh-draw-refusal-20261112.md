- **`bootstrap_summary`, `bootstrap_result`, `bootstrap_ci` and the default `simulate` now refuse Gamma and Binomial fits with two or more ordinary random-effect terms.**
  This is an explicit change: a call that used to return a table (or fail late) now stops with an error.
  The refused shapes are `(1 | g) + (0 + x | g)`, crossed or nested intercepts such as `(1 | g) + (1 | h)`,
  mixed shapes such as `(1 | g) + (0 + x | g) + (1 | h)`,
  and slope-only shapes such as `(0 + x | g) + (0 + z | g)`.
  None of them has a fresh random-effect draw. Until now the bootstrap quietly drew responses with the
  random effects set to zero, so the variance-component interval collapses.
  The message names the terms you fitted. For Gamma it points to `simulate(fit; data = data, re = :zero)`,
  which still works. For Binomial that call fails on this version (the fit does not carry trials),
  so the message says no draw is available yet.
  A single `(1 | g)` is unchanged for both families. Gamma `(0 + x | g)` and `(1 + x | g)` are refused too (see the separate note on Gamma slopes).
  Decision P11c: refuse until the fresh draw exists.
