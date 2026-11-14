- **`bootstrap_summary`, `bootstrap_result` and `bootstrap_ci` now refuse a Gamma fit with a single `(0 + x | g)` or `(1 + x | g)`.**
  This changes what the call computes, so it is announced here. These two shapes have no fresh random-effect draw.
  Until now the bootstrap quietly drew responses with the random effects set to zero, so the variance-component interval collapsed and a table came back as if nothing were wrong.
  The call now stops with an error that names the fitted term, for example `no fresh random-effect draw is implemented for Gamma() with a random slope without an intercept ((0 + x | g))`.
  The default `simulate(fit; data = data)` already stopped for these shapes with a general message; it now gives the same named message.
  `simulate(fit; data = data, re = :zero)` still works and gives the conditional draw, now only when you ask for it.
  A single `(1 | g)` is unchanged. Binomial single-slope fits are not changed by this entry.
