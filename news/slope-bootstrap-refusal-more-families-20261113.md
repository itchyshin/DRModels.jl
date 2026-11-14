- **Beta, Student and LogNormal random-slope bootstrap refuses by name.**
  `bootstrap_summary`, `bootstrap_result`, `bootstrap_ci` and the default
  `simulate(fit; data = data)` on a Beta or Student fit with a random slope
  (`(0 + x | g)`, `(1 + x | g)`), on a LogNormal fit with a correlated
  intercept and slope (`(1 + x | g)`), or on any of the three with two or more
  random-effect terms, fell back to drawing responses with the random effects
  at zero, which collapses the variance-component interval without a message.
  They now stop with the same message the Gamma and Binomial fits give, naming
  the fitted terms. A LogNormal fit with a single `(0 + x | g)` already draws
  new random effects and is unchanged. `simulate(fit; data = data, re = :zero)`
  still works for these fits. Fixed-effects-only fits and a single `(1 | g)`
  are unchanged.
