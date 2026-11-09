# Laplace objective on the quadrature routes (`marginal = :Laplace`), design and pre-registration, 2026-11-09

Reader: a DRModels.jl user who wants log-likelihoods identical to drmTMB's, and whoever owns the R-Julia parity lane. Purpose: say what is added, which of the 54 wave-8 log-likelihood disagreements it should close, and what "agrees with R" means, before any reference number for the new cells is read.

## Which integral to prefer

A random effect has to be integrated out of the likelihood. There are two ways to do it here.

- **Adaptive Gauss-Hermite quadrature** (the default on the older single-term and complete-SD routes) evaluates the integral at several points per group. It is the more accurate integral: in the wave-8 check against a 100-node reference it matched to 2e-3 or better in 36 of 36 one-dimensional cells.
- **Laplace approximation** (what drmTMB uses) replaces the integrand by a Gaussian at its peak. It is faster and smooth, but it is only approximate. With few observations per group and a non-Gaussian response it can sit above or below the exact value by 0.03 to 0.5 log-likelihood units. It is higher than exact for Poisson and Gamma slope-only and for Tweedie, and lower for most other cells.

If you want the most accurate value, keep the default. If you want a number that matches drmTMB (to compare fits across the two packages, to reproduce a drmTMB analysis, or to run a likelihood-ratio test against a drmTMB fit), pass `marginal = :Laplace`. The two values are not comparable with each other, so `lrtest` and AIC comparisons must not mix them. The point estimates are usually close either way. The default is unchanged.

## Change

The keyword that already exists for a plain `(1 | g)` (`marginal = :Laplace`) is accepted by the following routes, which today stop with "supports `marginal = :LA` only". No new keyword is introduced, and without it every fit is as before.

1. One independent term, intercept or slope, with or without `sd(g, coef(.)) ~ ...` formulas, on Poisson, NegBinomial2, Binomial, Beta, Gamma, Student and Tweedie: the dense TMB-convention Laplace fitter `_fit_indep_re_laplace` (already used for two or more independent terms) with one component.
2. Two or more independent terms with `sd()` formulas on Binomial: the same fitter (the older quadrature route stays the default).
3. Tweedie plain `(1 | g)` and `(0 + x | g)`, which refuse the keyword today.

## Cell classes (from the wave-8 report and the independent-terms pull request)

(i) a routing change reaches R's value; (ii) needs a Laplace objective for that route; (iii) differs for another reason (plateau, near-singular, boundary, different model), recorded and left. The per-cell table is `test/fixtures/laplace_parity/cells.tsv` and the counts are in the pull request.

## Pre-registered agreement rule

Reference: drmTMB 0.8.1, `engine = "tmb"`, Totoro library `~/drm_campaign_20261108/Rlib`, the committed gap-map data (`test/fixtures/nongauss_indep/data_<family>.csv`, 400 rows, 39 groups), generator `test/fixtures/laplace_parity/gen_ref.R`. A cell **agrees** when all hold at the optimum:

- `|logLik_Julia - logLik_R| <= 1e-4`;
- every fixed-effect coefficient (mean and, where present, log-sigma) differs by at most `1e-3` of R's standard error scale, that is `|diff| / se_R <= 1e-3`;
- every random-effect log SD (or SD-formula coefficient) differs by at most `5e-3`, with a term whose SD is at zero (log SD below -6) in either fit exempt from this criterion only.

These limits were fixed before any reference value for the new cells was read, and are not to be widened afterwards. A cell that misses one is investigated, not accepted. R fits with `is_converged() = FALSE` are recorded and not counted as agreement or failure.

The default-behaviour test fits three existing cells with no keyword on this branch and compares the stored log-likelihoods, coefficients and `converged` flags with values recorded on the base commit; they must be bit-identical.
