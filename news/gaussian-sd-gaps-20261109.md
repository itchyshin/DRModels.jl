- **Gaussian ordinary and crossed random-effect shapes that R fits.** Four
  shapes that the Gaussian mean route refused now fit by exact dense maximum
  likelihood (ML only; `method = :REML` is refused with a message that says so, except that a single labelled q2 block `(1 + x | p | g)` is the ordinary correlated block and fits under REML):
  an unstructured block of three or four coefficients with no `sd()` formula,
  `(1 + x + z | g)`; a block label `(1 + x | p | g)` when only one term carries
  the label and `sigma` has no random effect; a correlated block beside random
  intercepts on other groupings, `(1 + x | g) + (1 | h)`, with optional
  `sd(g, coef(·)) ~ …` per block coefficient and `sd(h) ~ …`; and independent
  intercept and slope terms with `sd(g, coef(·)) ~ …`,
  `(1 | g) + (0 + x | g)`. Agreement with drmTMB 0.8.1 on three committed
  datasets is in `docs/dev-log/2026-11-09-gaussian-sd-gaps.md`. Still refused:
  a mean `sd(g) ~ …` formula together with a `sigma` random effect, and any
  correlated block beside a slope term.
