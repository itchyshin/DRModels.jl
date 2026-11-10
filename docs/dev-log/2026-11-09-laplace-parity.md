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

The default-behaviour test fits three existing cells with no keyword on this branch and compares the stored log-likelihoods, coefficients and `converged` flags with values recorded on the base commit; they must match to a relative 1e-9 (bit-identical on the Mac; the last bit differs on Linux).

## Result (written after the run; the rule above was not changed)

Reference runs: drmTMB 0.8.1 on Totoro, library `~/drm_campaign_20261108/Rlib`, one core per fit, `nice -n 10`; new fits for the single `(1 | g)` and correlated-block cells by `test/fixtures/laplace_parity/gen_ref.R`, the slope and independent-term references come from `test/fixtures/nongauss_indep/`. Julia fits: `tools/laplace_parity_compare.jl`, Julia 1.10.12 on the Mac. Gaps are Julia against drmTMB.

Step 1 classes of the 54 wave-8 disagreement cells: (i) 38, (ii) 11, (iii) 5. In 42 cells Julia's quadrature value is higher than drmTMB's Laplace value, in 12 it is lower.

With `marginal = :Laplace`:

- **44 of 54 agree** under the rule. The worst gaps over those 44 are logLik 8e-9, fixed effects 1.2e-7 of R's standard error and log SD 1.7e-7, against limits of 1e-4, 1e-3 and 5e-3. They are: the 6 plain `(1 | g)` cells that already worked, 29 single-term and independent-term cells, 2 Tweedie `sd-none` cells, and 7 correlated-block cells (Binomial q2 `sd-one`, `sd-first`, `sd-last`, `sd-z`, plain Poisson, NB2 and Binomial q2).
- **1 cell has matching numbers but Julia flags it.** Binomial q3 `sd-first`: logLik 3e-8, fixed effects 1e-8 and log SD 3e-8 from drmTMB, but the fit reports `converged = false`.
- **1 cell differs.** Binomial q4 `sd-first` stops unconverged 0.013 logLik below drmTMB (fixed effects 3e-4 and log SD 4e-4 away). I did not find out whether a restart reaches R's point.
- **3 cells stay on a ridge (class iii).** Binomial independent `sd-z` q2, q3 and q4: drmTMB reports `is_converged() = FALSE` at a boundary point with intercept SD coefficients near -54 and +57; the dense Laplace route stays 0.45, 0.47 and 1.47 below and reports `converged = false`, as the independent-terms pull request found for `sd-first`.
- **5 cells are not changed (class iii).** Two Binomial spatial cells where drmTMB stopped early, two Binomial `animal` cells and one Gaussian spatial cell where Julia fits an extra parameter on purpose.

Two things the keyword reuses rather than adds: Binomial correlated blocks use the SD-regression group integral with a single adaptive node, which is the Laplace approximation; counts use the identity-kernel Laplace of the structured q = 2 routes.

Defaults are unchanged: `test/test_laplace_parity.jl` fits three cells without the keyword and compares logLik and every coefficient with the values recorded on the base commit (relative 1e-9; bit-identical on the Mac, the last bit differs on Linux). Six existing test files asserted that shapes now fitted were refused (`test_gamma_sd_regression.jl`, `test_student_sd_regression.jl`, `test_ordinary_laplace.jl`, `test_binomial_sd_public.jl`, `test_binomial_sd_regression.jl`, `test_sigma_re_laplace.jl`) and were updated to the new behaviour; the wording of the refusals that remain still names `marginal = :LA`.

### All 54 cells

| # | Cell | Step 1 class | Laplace fit vs drmTMB | dLogLik |
| --- | --- | --- | --- | --- |
| 1 | `binomial\|spatial\|lab0\|q2\|sd-z\|cm0\|ML` | iii | R stopped early (R pdHess FALSE); not changed |  |
| 2 | `binomial\|indep\|lab0\|q2\|sd-z\|cm0\|ML` | i (to test) | ridge: R non-converged, Julia 0.45 below | 4.5e-01 |
| 3 | `binomial\|spatial\|lab0\|q1\|sd-z\|cm0\|ML` | iii | R stopped early; not changed |  |
| 4 | `binomial\|animal\|lab0\|q2\|sd-one\|cm0\|ML` | iii | Julia estimates an extra correlation; not changed |  |
| 5 | `binomial\|indep\|lab0\|q3\|sd-z\|cm0\|ML` | i (to test) | ridge: R non-converged, Julia 0.47 below | 4.7e-01 |
| 6 | `gaussian\|spatial\|lab0\|q1\|sd-none\|cm0\|ML` | iii | Julia estimates the spatial range; not changed |  |
| 7 | `binomial\|animal\|lab0\|q2\|sd-z\|cm0\|ML` | iii | Julia estimates an extra correlation; not changed |  |
| 8 | `binomial\|ordinary\|lab0\|q4\|sd-first\|cm0\|ML` | ii | Julia stops unconverged, 0.013 below | 1.3e-02 |
| 9 | `binomial\|indep\|lab0\|q4\|sd-z\|cm0\|ML` | i (to test) | ridge: R non-converged, Julia 1.47 below | 1.5e+00 |
| 10 | `binomial\|ordinary\|lab0\|q3\|sd-first\|cm0\|ML` | ii | values agree to 3e-8, Julia flags converged = false | 3.1e-08 |
| 11 | `binomial\|ordinary\|lab0\|q2\|sd-none\|cm0\|ML` | ii | agrees | 4.3e-13 |
| 12 | `binomial\|ordinary\|lab0\|q2\|sd-one\|cm0\|ML` | ii | agrees | 4.5e-13 |
| 13 | `binomial\|ordinary\|lab0\|q2\|sd-first\|cm0\|ML` | ii | agrees | 8.5e-13 |
| 14 | `binomial\|ordinary\|lab0\|q2\|sd-z\|cm0\|ML` | ii | agrees | 7.4e-13 |
| 15 | `binomial\|ordinary\|lab0\|q2\|sd-last\|cm0\|ML` | ii | agrees | 8.2e-09 |
| 16 | `binomial\|indep\|lab0\|q2\|sd-one\|cm0\|ML` | i | agrees | 5.1e-13 |
| 17 | `beta\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 2.1e-11 |
| 18 | `beta\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | i (existing) | agrees | 2.2e-11 |
| 19 | `nb2\|ordinary\|lab0\|q2\|sd-none\|cm0\|ML` | ii | agrees | 1.6e-12 |
| 20 | `beta\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 1.2e-11 |
| 21 | `poisson\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 0.0e+00 |
| 22 | `poisson\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 6.8e-11 |
| 23 | `gamma\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 3.4e-11 |
| 24 | `gamma\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 6.8e-13 |
| 25 | `student\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 3.9e-12 |
| 26 | `student\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | i (existing) | agrees | 1.0e-11 |
| 27 | `student\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 7.2e-12 |
| 28 | `beta\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 8.7e-13 |
| 29 | `student\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 3.5e-11 |
| 30 | `student\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 2.0e-11 |
| 31 | `beta\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 8.5e-13 |
| 32 | `binomial\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 4.0e-13 |
| 33 | `binomial\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 1.1e-13 |
| 34 | `nb2\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 3.3e-12 |
| 35 | `nb2\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | i (existing) | agrees | 4.2e-12 |
| 36 | `nb2\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 5.1e-10 |
| 37 | `binomial\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | i (existing) | agrees | 2.3e-13 |
| 38 | `binomial\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 4.0e-13 |
| 39 | `binomial\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 6.5e-13 |
| 40 | `tweedie\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 6.4e-11 |
| 41 | `tweedie\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | ii | agrees | 1.7e-12 |
| 42 | `tweedie\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 1.7e-12 |
| 43 | `poisson\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 3.7e-10 |
| 44 | `poisson\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 3.5e-11 |
| 45 | `gamma\|ordinary\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 2.7e-09 |
| 46 | `gamma\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 5.7e-11 |
| 47 | `poisson\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | i (existing) | agrees | 3.7e-10 |
| 48 | `gamma\|ordinary\|lab0\|q1\|sd-none\|cm0\|ML` | i (existing) | agrees | 2.7e-09 |
| 49 | `nb2\|ordinary\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 2.0e-11 |
| 50 | `tweedie\|slope\|lab0\|q1\|sd-none\|cm0\|ML` | ii | agrees | 1.2e-11 |
| 51 | `tweedie\|slope\|lab0\|q1\|sd-one\|cm0\|ML` | i | agrees | 1.1e-11 |
| 52 | `nb2\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 1.9e-12 |
| 53 | `poisson\|ordinary\|lab0\|q2\|sd-none\|cm0\|ML` | ii | agrees | 3.5e-10 |
| 54 | `tweedie\|slope\|lab0\|q1\|sd-z\|cm0\|ML` | i | agrees | 1.5e-12 |

## Not covered

Binomial q4 `sd-first` and the three independent `sd-z` cells (above). Correlated blocks with a non-Binomial family and three or more coefficients are refused under `marginal = :Laplace`; none were in the 54. Only one data set and seed per cell, as in the gap map.

## Test suites

Full suites (16 shards per Julia version, one thread each) are running on Totoro at the time of writing; results are in the pull request description.
