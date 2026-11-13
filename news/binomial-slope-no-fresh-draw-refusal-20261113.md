- **`bootstrap_summary`, `bootstrap_result` and `bootstrap_ci` now refuse a Binomial fit whose random effects are anything other than one `(1 | g)`.**
  This changes what the call computes, so it is announced here. The refused shapes are a single `(0 + x | g)` or `(1 + x | g)`, an intercept bar plus a slope bar on the same group such as `(1 | g) + (0 + x | g)`, and crossed bars such as `(1 | g) + (1 | h)` (Laplace and AGHQ). These shapes have no fresh random-effect draw.
  Where the bootstrap used to run on them, it quietly drew responses with the random effects set to zero, so the variance-component interval collapsed and a table came back as if nothing were wrong. For `(1 + x | g)` that was already the case; for the other shapes it is what would happen as soon as the fit stores its trial counts.
  The call now stops with an error that names the fitted terms, for example `no fresh random-effect draw is implemented for Binomial() with a random slope without an intercept ((0 + x | g))`.
  The default `simulate(fit; data = data)` stops with the same named message for these shapes. `simulate(fit; data = data, re = :zero)` is not changed.
  Fixed-effects-only fits and a single `(1 | g)` are unchanged.
  This is a maintainer decision: refuse until the fresh draw exists.
