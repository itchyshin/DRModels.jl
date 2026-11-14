- **Beta, Student and LogNormal random-slope bootstrap refuses by name.**
  `bootstrap_summary`, `bootstrap_result`, `bootstrap_ci` and the default
  `simulate(fit; data = data)` on a Beta, Student or LogNormal fit with a
  random slope (`(0 + x | g)`, `(1 + x | g)`), or with two or more
  random-effect terms, fell back to drawing responses with the random effects
  at zero, which collapses the variance-component interval without a message.
  They now stop with the same message the Gamma and Binomial fits give, naming
  the fitted terms. `simulate(fit; data = data, re = :zero)` still works for
  these fits. Fixed-effects-only fits and a single `(1 | g)` are unchanged.
