- **`bootstrap_summary` and the default `simulate` now refuse Gamma and Binomial `(1 | g) + (0 + x | g)`.**
  These two shapes have no fresh random-effect draw. Until now the bootstrap quietly drew responses with
  the random effects set to zero and returned a table whose variance-component interval collapses.
  It now stops with a message that names the shape; `simulate(fit; data = data, re = :zero)` still
  works. A single `(1 | g)` is unchanged. Decision P11c: refuse until the fresh draw exists.
