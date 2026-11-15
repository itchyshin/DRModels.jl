- **Crossed Tweedie: a power at the end of its search interval is flagged.**
  The crossed-intercept Tweedie fit searches the power `p` on the logit scale
  inside [-6, 6] (p between 1.0025 and 1.9975). When the fitted power is within
  1e-5 of either end (measured in p), the fit now warns once, names the power and the
  interval, and sets `fit.geometry` to `:power_at_bound`. The value is a bound,
  not an interior estimate. `is_converged` is already false for every crossed Tweedie
  fit and does not change. The interval, the search and every estimate are
  unchanged; fits with an interior power are bit-identical to before. This is the warning only;
  a smooth map of the power is a separate decision. The warning says what to do next:
  `drm` has no option to fix the power or widen the interval, so refit with `Poisson()`
  or `Gamma()` when the response looks like one of those.
