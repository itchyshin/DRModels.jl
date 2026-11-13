- **Crossed Tweedie: a power at the end of its search interval is flagged.**
  The crossed-intercept Tweedie fit searches the power `p` on the logit scale
  inside [-6, 6] (p between 1.0025 and 1.9975). When the search ends within
  0.001 of either end, the fit now warns once, names the power and the
  interval, and sets `fit.geometry` to `:power_at_bound`, so `is_converged` is
  false and the R bridge reports `converged = false`. The value is a bound,
  not an interior estimate. The interval, the search and every estimate are
  unchanged; fits with an interior power are bit-identical to before.
