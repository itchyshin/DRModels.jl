# Spatial-kernel audit: DRModels.jl versus drmTMB (2026-11-08)

Branch `claude/spatial-kernel-audit-20261108`, on top of `claude/gamma-spatial-kernel-20261108`
(`a446857f3`, itself on draft PR #1268). Draft only; not merged.

## What drmTMB does

drmTMB has no estimated-range coordinate-spatial route. Every `spatial(..., coords = )` term, in
every family that admits it, goes through `drm_spatial_coords_precision()`
(`R/drmTMB.R`): correlation `exp(-d / r)` with `r` = the median positive pairwise distance (the
maximum if that median is not positive), plus `1e-6` on the diagonal. The range is FIXED. Bivariate
q=2 and q=4 spatial use the same function. The R-to-Julia bridge converts Gaussian `coords` to a
fixed `K` with it, so bridged fits already agree; this audit concerns DIRECT Julia `coords =` input.
drmTMB has no binomial spatial route (`Structured-effect syntax is planned, not implemented`).

Native kernel in Julia: `_gaussian_spatial_sd_kernel(coords, G, grp)` (`src/gaussian_lss.jl`).

## Audit table

"Native" = already `exp(-d / median positive d) + 1e-6 I`, fixed range. "Legacy" = mean off-diagonal
distance, `1e-8` ridge, fixed range. "Estimated" = log range is a free parameter.

| # | Julia route (file:line) | Julia kernel before | drmTMB counterpart | Decision |
|---|---|---|---|---|
| 1 | Gaussian q=1 intercept `spatial(1 \| g)` (`_fit_spatial_gaussian`, `gaussian_structured.jl:562`, `gaussian_core.jl:1727`); bootstrap `inference.jl:3930` | Estimated range, `exp(-d/rho)+1e-8 I` | Fixed native kernel | Different model by design (documented in `capabilities.md`). Left alone. The bootstrap uses the fitted range, so it stays in sync with the fit. |
| 2 | Gaussian SD regression, slope `spatial(0+x \| g)` and intercept `sd(g, spatial) ~ z` (`gaussian_lss.jl:715`) | Native | Fixed native | Aligned already. |
| 3 | Gaussian `sigma` structured / `sigma` SD (`gaussian_sigma_sd_laplace.jl:391`, `sigma_structured_laplace.jl:212`) | Native | Fixed native | Aligned already. |
| 4 | Gaussian, Poisson, NB2, Binomial q=2 `spatial(1 + x \| g)` SD (`_structured_q2_kernel`, `structured_q2_phylo_spatial.jl:43`; `twofield_sd_laplace.jl:512`; `binomial_sd_regression.jl:1101`; `inference.jl:3685`) | Native | Fixed native | Aligned already. |
| 5 | NB2 slope SD `spatial(0+x \| g)` (`negbinomial.jl:519`) | Native | Fixed native | Aligned already. |
| 6 | Gamma q=1, SD and slope (`gamma.jl:167,669,678`; bootstrap `inference.jl:3864`) | Native (previous commit) | Fixed native | Aligned; agreement 1e-7 R-SE units. |
| 7 | **Bivariate Gaussian q=2** `spatial(1 \| site)` (`gaussian_bivariate.jl:821`); bivariate LogNormal forwards here | Legacy | Fixed native | **Changed to native** when `spatial_range` is not supplied. |
| 8 | **Bivariate Gaussian q=4** (`gaussian_bivariate.jl:1046` kernel; `:957`, `:1188` recorded range via `_q4_default_spatial_range`) | Legacy | Fixed native | **Changed to native** by default. The recorded `fit.ranef.spatial_range` is now the median rule, and the helper accepts a 1-D vector. |
| 9 | Poisson q=1 `spatial(1 \| g, coords)` (`_fit_poisson_spatial_coord`, `poisson.jl:1587`) | Estimated range, unit-diagonal jitter kernel | Fixed native | Different model by design. Left alone. |
| 10 | Poisson / NB2 / Gamma / Beta / ZOB / Tweedie `spatial(1 \| g)` with `K` (`_poisson_structured_cov`, `poisson.jl:1156`) | User matrix | n/a | Nothing to align: no coordinate kernel is built. |
| 11 | ZOB, Tweedie, LogNormal, Student, beta-binomial, cumulative with `coords` | No Julia `coords` route (error, or `K` only) | drmTMB has some of these (Student, ZOB, positive-continuous `mu`) | Capability gap, not a kernel mismatch. Recorded; not addressed here. |
| 11a | **Beta** `spatial(1 \| g)` with `sd(g, spatial) ~ z`, and `spatial(1 + x \| g)` SD blocks (`beta.jl`, via `_structured_q2_kernel`, `structured_q2_phylo_spatial.jl:43`) | n/a (route added on this integration branch) | Fixed native | Takes `coords =` and builds the native kernel through `_gaussian_spatial_sd_kernel`; aligned by construction. Plain mean-only Beta `spatial(1 \| g)` with `coords` still goes through `_poisson_structured_cov` (as row 9) and is not part of this audit. |
| 12 | **Binomial** q=1 intercept, SD regression and slope `spatial(0+x \| g)` (`binomial_sd_regression.jl:1206`; bootstrap `inference.jl:2650`) | Legacy | **None** (drmTMB: not implemented) | Left on the legacy kernel, now behind an explicit `legacy_mean_range = true`. See the Binomial note. |
| 13 | Binomial mesh SD regression (`binomial_sd_regression.jl:2125`) | Mesh/SPDE | Mesh is Gaussian-only in drmTMB | Out of scope (not a coordinate kernel). |

Routes 7 and 8 share `_spatial_covariance_from_coords`; an explicit `spatial_range = r` keeps the
old behaviour (`exp(-d/r) + 1e-8 I`) because it is a user-chosen different model. The Binomial
callers pass `legacy_mean_range = true`, so their behaviour is byte-identical to before.

## Binomial and `test_binomial_spatial_sd_regression` on Julia 1.12

The failing assertion is the probit slope check, `abs(coef(fit, :sd)[2] - 0.4) < 0.45`, which
gives 0.5411 on Julia 1.12 before and after this branch (the Binomial routes are unchanged here).

To test whether the kernel is the cause, the Binomial routes were switched to the native kernel in a
scratch copy (not committed) and the original test was run: every assertion passes (23 + 22 + 1), but the data
change, because the test simulates its own data from the kernel. A fixed seed is a single draw, so
that proves little. Twenty seeds per kernel, with the data-generating kernel equal to the fitted
kernel in each arm (Julia 1.12, `g_tol = 1e-5`, slope route, `alpha2 = 0.4`):

| kernel (data and fit) | link | converged | \|alpha2_hat - 0.4\| < 0.45 | median error | max error |
|---|---|---|---|---|---|
| legacy | probit | 20/20 | 17/20 | 0.072 | 2.01 |
| native | probit | 20/20 | 18/20 | 0.083 | 1.70 |
| legacy | logit | 20/20 | 19/20 | 0.134 | 10.67 |
| native | logit | 20/20 | 19/20 | 0.138 | 1.67 |

Reading: the 0.45 bound fails on roughly 1 draw in 7 under either kernel, with occasional outlying
fits. The seeded probit failure is fixture luck, not the kernel, so the kernel switch is NOT a fix
for it. This branch does not touch that test's tolerance or seed. Separately, the Binomial q=1
route disagrees with the Binomial q=2 route (native) inside DRModels.jl; whether to move Binomial to
the native kernel for internal consistency, although drmTMB has no counterpart, is a decision for the
owner (a two-line change: drop `legacy_mean_range = true` at the two call sites and in the test).

## R versus Julia agreement at the optimum

Thresholds were fixed before any Julia result was read. A dataset AGREEs iff all of:
max absolute difference <= 1e-3 and max difference <= 0.01 R-SE units (betas, log sigma, rho12
link, log random-effect SD); |loglik difference| <= 1e-3; max |correlation difference| <= 5e-3
(natural scale). A dataset is scored only if R converged, Julia converged, and all fitted
correlations satisfy |r| < 0.99 on both sides. Data are simulated in R from a spatial field with an
exponential kernel of range 0.7 x mean distance (so neither kernel is the truth), committed in
`test/data/spatial_kernel_audit/`. Julia 1.12.7 on Totoro (1 thread); R is drmTMB with
`engine = "tmb"`. q=4 used `q4_g_tol = 1e-6` (and 1e-10, with identical loglik).

| dataset | route | kernel | max abs diff | max diff (R-SE units) | max abs cor diff | loglik diff (J-R) | verdict |
|---|---|---|---|---|---|---|---|
| q2a | q=2 | old (mean, 1e-8) | 3.84e-02 | 1.81e-01 | 6.68e-04 | 2.05e-01 | DISAGREE |
| q2a | q=2 | native (median, 1e-6) | 7.35e-07 | 7.65e-06 | 2.56e-06 | 4.92e-10 | AGREE |
| q2b | q=2 | old (mean, 1e-8) | 2.17e-02 | 9.11e-02 | 2.41e-04 | 1.03e-01 | DISAGREE |
| q2b | q=2 | native (median, 1e-6) | 2.05e-06 | 4.75e-06 | 1.91e-06 | 3.70e-10 | AGREE |
| q2c | q=2 | old (mean, 1e-8) | 1.48e-01 | 6.00e-01 | 2.60e-02 | 5.66e-01 | DISAGREE |
| q2c | q=2 | native (median, 1e-6) | 1.57e-05 | 6.38e-05 | 5.19e-06 | -1.66e-09 | AGREE |
| q4a | q=4 | old (mean, 1e-8) | 1.63e-02 | 1.22e-01 | 3.56e-03 | -1.13e-01 | DISAGREE |
| q4a | q=4 | native (median, 1e-6) | 2.25e-03 | 4.67e-03 | 1.17e-03 | -2.38e-05 | DISAGREE |
| q4b | q=4 | old (mean, 1e-8) | 3.99e-02 | 2.91e-01 | 8.58e-03 | 9.33e-02 | DISAGREE |
| q4b | q=4 | native (median, 1e-6) | 1.75e-03 | 6.49e-03 | 4.57e-04 | -2.75e-05 | DISAGREE |
| q4c | q=4 | old (mean, 1e-8) | 1.83e-01 | 1.34e+00 | 8.42e-03 | 1.15e+00 | DISAGREE |
| q4c | q=4 | native (median, 1e-6) | 4.72e-03 | 6.77e-03 | 2.17e-03 | -8.60e-04 | DISAGREE |

All six datasets were scored (R converged, Julia converged, no fitted correlation at the boundary).
q=2: every dataset DISAGREES before and AGREES after, at 7e-7 to 1.6e-5 absolute, <= 6.4e-5 R-SE units.
q=4: error falls 26x to 200x in R-SE units (0.12 to 1.34 before, <= 0.0068 after) and the loglik
difference falls from about 0.1 to <= 9e-4, but every q=4 dataset still misses the pre-registered
ABSOLUTE bound of 1e-3 (1.8e-3 to 4.7e-3), although each meets the R-SE, loglik and correlation
bounds. Tightening Julia's q=4 tolerance from 1e-6 to 1e-10 gives the identical loglik to 12 digits, so
the run had not been limited by `q4_g_tol`; Julia's loglik sits 2e-5 to 9e-4 BELOW R's. This was
traced afterwards to Julia's default `f_reltol = 1e-7` stopping rule: the q=4 L-BFGS stops on a relative
objective change of 1e-7, which on these flat ridges leaves it up to about 1.2e-3 short of the optimum,
whatever `q4_g_tol` is. R is already at its optimum (re-optimising R with tolerances of 1e-15 moves its
loglik by at most 3e-7). Passing `q4_f_reltol = 1e-12` brings Julia to R to 4e-5 in loglik
(`test/test_spatial_kernel_audit.jl`). So the residual is Julia stopping early at the default tolerance,
not R stopping short and not a kernel difference. The committed q=4 test keeps a looser bound for the
default tolerance and tests the tight tolerance separately.


## Reproducing

`test/data/spatial_kernel_audit/r_gen_fit.R` regenerates the data and R fits.
`test/test_spatial_kernel_audit.jl` refits in Julia and asserts the above against the committed R
values (plus a control: the old mean-distance rule, via an explicit `spatial_range`, is
measurably off R on the same data).
