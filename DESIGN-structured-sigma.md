# Design: NB2 / Gamma structured sigma random effects

Branch `claude/nb2-gamma-structured-sigma-20261108` (from `claude/labelled-ordinary-poisson-nb2-20261108`).
Code: `src/sigma_structured_laplace.jl`; the Gaussian structured-sigma spec builder in
`src/gaussian_sigma_sd_laplace.jl` was factored into `_gsl_structured_sigma_spec` and is shared.
Test: `test/test_nb2_gamma_structured_sigma.jl`. R agreement: drmr repo,
`docs/dev-log/parity-crosswalk-evidence-20261107/wave4-structured-sigma/`.

## What native drmTMB 0.8.1 fits (probed 2026-11-08, ML, installed library 0.8.1)

Data: 40 groups x 12 rows, sigma-side structured effect, providers `animal(A=)`, `relmat(K=)`,
`phylo(tree=)`, `spatial(coords=)`. Refusals are verbatim from R.

| Shape | NB2 | Gamma |
| --- | --- | --- |
| `sigma ~ 1 + P(1 + x \| id)`, no SD formula | **fits** (all four providers) | refused (1) |
| `sigma ~ 1 + P(1 \| id)`, with or without `sd(id, level, dpar = "sigma") ~ z` | refused (2) | refused (1) |
| `sigma ~ 1 + P(1 + x \| id)` plus `sd(..., dpar = "sigma", coef = ...) ~ z` | refused (3) | refused (1) |
| `sigma ~ 1 + P(0 + x \| id)` | refused (2) with `Requested structured coefficient: "x"` | refused (1) |
| `mu ~ x + phylo(1 \| id)` plus `sigma ~ 1 + (1 \| id)` with `sd(id, dpar = "sigma") ~ zc` | refused (4) | not probed |

(1) `Gamma structured effects are admitted only in `mu` in this route. / The `sigma` formula contains structured provider: "<P>". / Keep `sigma` fixed-effect and move one unlabelled q=1 `phylo()` or `relmat()` intercept to `mu`.`

(2) `NB2 structured `sigma` effects currently support only unlabelled intercept-plus-one-slope structured terms. / Requested structured coefficient: "(Intercept)". / Use `animal(1 + x | id, Ainv = Ainv)` for this count structured-scale gate; multiple slopes, labelled covariance, q2/q4, and structured count location-scale blocks remain planned.`

(3) `NB2 structured SD regression currently supports one animal, phylogenetic, related-matrix or spatial location coefficient, or an `animal(1 + x | id)` intercept plus one slope with one SD formula per coefficient. / Use one q1 `animal()` or `phylo()` location term with a matching structured `sd(group, level = ...)` formula; remove other random, structured, sigma, or zero-inflation terms.`

(4) `NB2 `sigma` random intercepts cannot be combined with `mu` random effects in this first gate.`

Also confirmed: the Gaussian shapes `sigma ~ 1 + phylo(1 | id)` / `animal(1 | id)` with an SD formula fit
natively (control), and ordinary NB2/Gamma `sd(id, dpar = "sigma") ~ zc` fits natively (already in Julia).

So **no native drmTMB route fits an SD formula on a structured sigma effect for NB2 or Gamma**.
Consequently there is no R twin for the requested headline shape.

## What is implemented

* **Shape A (R fits it; NB2 only).** `y ~ x`, `sigma ~ 1 + P(1 + x | g)`.
  `eta_i = X_sigma beta + s1 u1[g] + s2 x_i u2[g]`, `u1, u2 ~ N(0, K)` independent, `s1, s2` constant.
  `theta = [beta_mu; beta_sigma; log s1; log s2]`; blocks `:mu`, `:sigma`, `:sigma_sd_intercept`,
  `:sigma_sd_<slope>` (names avoid `sdtarget_` because no `sd()` formula exists to resolve).
* **Shape B (JULIA-ONLY; R refuses it).** `y ~ x`, `sigma ~ 1 + P(1 | g)`, `sd(g, level, sigma) ~ z`.
  `eta_i = X_sigma beta + exp(Z alpha)[g] u[g]`, `u ~ N(0, K)`. Same engine as shape A with one field;
  the model and kernels are those of the Gaussian structured-sigma route (6b), so this is a family
  swap of an already-verified construction, but with no native comparator.
* K: raw `A`/`K` (animal/relmat, `ids =` label matching works), tree correlation (phylo),
  native fixed kernel `exp(-d/median positive d) + 1e-6 I` (spatial).
* Laplace value, inner damped Newton, exact one-step implicit gradient, boundary polish, multimodality
  check, Wald covariance from central differences of the analytic gradient: as the Gaussian engine
  (`_fit_gaussian_sigma_sd_laplace`; shared `_gsl_kernel`, `_gsl_mode`, `_ordinary_laplace_optimize`).
  Shape A/B use the new multi-field `_fit_sigma_fields_laplace` (sigma-only latent, mean fixed).
* Cap: 600 latent coordinates (`fields * levels`).

## Gaps (not implemented, by design)

Gamma shape A (R refuses; nothing to compare, and no stated need); sigma slope with an SD formula;
mean random/structured effect together with a structured sigma effect; labelled correlated
`(1 + x | p | g)` on sigma; `Ainv`; REML; `zi`/`hu`; missing responses; spatial range estimation
(the kernel is the native fixed one); coefficient-selected SD (`coef(...)`) on shape B.
`simulate(fit; re = :zero)` draws at the fitted conditional sigma (`scales[:sigma]`), the existing
convention of the Gaussian and joint-sigma fits, not at sigma_fixed.
