- **`vc` and `re_sd` on a correlated slope fitted with `marginal = :Laplace`.**
  `(1 + x | g)` fitted with `marginal = :Laplace` stores the block as two log
  SDs and one `atanh` correlation, not as a log-Cholesky factor. `vc` and
  `re_sd` read the Binomial block as a Cholesky factor, so they returned a
  wrong slope SD and correlation (they now agree with drmTMB 0.8.1 on four
  data sets per family). For Poisson and NegBinomial2 they refused with a
  message about `sd(group) ~ ...`; they now return the SDs and the covariance.
  A Binomial `(1 + x + x2 | g)` block (stored as log SDs and partial
  correlations, on the default and the Laplace routes) was also read as a
  Cholesky triple; `vc` and `re_sd` now refuse it with a message that points
  to `coef(fit, :recov)`. The printed heading of these blocks no longer says
  "Cholesky". Fits on other routes return the same values as before.
