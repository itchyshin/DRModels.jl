- **More Gaussian random-effect shapes that R fits.** Two further groups fit by
  ML and agree with drmTMB 0.8.1 at the optimum on committed datasets (log-likelihood
  within 5e-9 on 30 fits over three datasets; evidence and refusals in
  `docs/dev-log/2026-11-09-gaussian-sd-groupf.md`). First, a correlated
  `(1 + x | g)` block beside random slope terms: `(1 + x | g) + (0 + z | h)`,
  two blocks `(1 + x | g) + (1 + z | h)`, and a block beside
  `(1 | h) + (0 + z | h)` or `(0 + z | g)`, with optional
  `sd(·, coef(·)) ~ …` formulas (dense exact likelihood, 5,000-row cap). Second, a mean
  `sd(g) ~ …` formula together with a `sigma` random intercept, independent
  or as the labelled pair `(1 | p | g)` in `mu` and `sigma` (dense Laplace).
  Still refused, by name: an `sd()` formula on a grouping that carries a block and
  another term (drmTMB refuses it too), a 3- or 4-coefficient block beside a slope
  (drmTMB's own fit does not converge), a label shared by two terms (the message now
  says drmTMB ignores such a label), and REML on all of these.
