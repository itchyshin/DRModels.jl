# REML for the Gaussian structured sigma intercept (`sd(g, level, sigma)` route)

Status: implemented in `_fit_gaussian_sigma_sd_laplace_reml` (`src/gaussian_sigma_sd_laplace.jl`).
Test: `test/test_gaussian_structured_sigma_reml.jl`.

## Gap

`drm(bf(y ~ x, sigma ~ 1 + KIND(1 | g), sd(g, LEVEL, sigma) ~ z), Gaussian(); method = :REML)` with
KIND in animal / relmat / phylo / spatial threw "supports maximum likelihood only", while native
drmTMB (`REML = TRUE`, v0.8.1) fits it. Scope here: fixed-effect mean, ONE structured sigma
intercept with an SD formula (the shapes `_gsl_structured_sigma_spec` admits). Ordinary sigma
random effects, mean random effects and NB2 / Gamma stay ML-only.

## What native R does (read-only: `R/drmTMB.R`, `drm_apply_estimator_spec`, `drm_validate_reml_spec`)

There is no closed-form restriction. When a sigma variance component exists, R appends `beta_mu`
and `beta_sigma` to TMB's random-effect names (`tmb_random_names`), so TMB's Laplace step runs over
all of `w = (v, beta_mu, beta_sigma)` with FLAT priors on the two coefficient blocks. The SD-formula
coefficients `alpha` stay outer parameters. R flags the fit `estimator_exact = FALSE` because the
sigma integral is a Laplace one (the density is not Gaussian in `beta_sigma`).

## Objective

With `F(w)` the ML joint negative log density in the standardised latent `v ~ N(0, K)` (as in the
ML route), `H = d2F/dw2` over ALL of `w`, `w^` the joint inner mode and `p = p_mu + p_sigma`:

    nll_R(alpha) = F(w^) + 1/2 log det H(w^) + 1/2 log det K - (p/2) log 2 pi

* Fixed-effect integration term: `+1/2 log det` of the `beta` rows/columns of `H` enters through
  the full `H` (the Schur-complement form `1/2 log det H_vv + 1/2 log det S_beta`, with the sigma
  coefficients coupled to `v` through `d_i`), plus the Laplace constant `-(p/2) log 2 pi` of the
  flat coordinates. The `v` block's `2 pi` constants cancel against the prior exactly as in ML.
* `loglik` reports `-nll_R` (so `reml_loglik == loglik`); `ml_loglik` is the ML Laplace value at
  the REML parameters. REML values are comparable across SD/sigma structures only, never against ML.
* Linear reparametrisation `u = s v` leaves the Laplace value invariant, so R's raw-effect and this
  standardised parametrisation agree (confirmed by the REML agreement below).
* Gradient in `alpha`: ForwardDiff of `nll_R` with `w^` replaced by one Newton step from the
  converged Float64 mode (exact: the inner gradient vanishes there).

## Reported coefficients and covariance

Coefficients are `(beta_mu^, beta_sigma^, alpha^)`; `beta^` is the joint-mode value at `alpha^`.
Covariance follows TMB's `sdreport` for random effects:

    Cov(beta) = [H^-1]_beta + J_beta V_alpha J_beta',   Cov(beta, alpha) = J_beta V_alpha,
    J = d w^ / d alpha  (implicit function, one differentiated Newton step),

with `V_alpha` the inverse finite-difference Hessian of the analytic `alpha` gradient.

## What is estimable / not claimed

* `alpha` (SD coefficients) and `beta` are estimable as in ML; REML debiases the SD, as in R.
* No exactness claim for the sigma side (R says `estimator_exact = FALSE`); no claim about small-g
  coverage, boundary inference, or the mean-side `beta_mu` interval (R's `confint` is unavailable
  for REML-integrated coefficients).
* Dense engine: `p + G <= 600` latent coordinates (same cap as ML).
* Multi-modal inner problem: same guard as ML (second PD mode from a prior start marks the fit
  not converged).

## Evidence (R = drmTMB 0.8.1 `REML = TRUE`; five StableRNG fixtures, G = 32, n = 320)

Thresholds fixed before fitting: |dlogLik| < 1e-6; coefficients within 1e-4 R-SEs; SE ratio
within 1e-3; independent dense oracle 1e-8. Measured: max |dlogLik| 2.7e-12; max |dcoef| 3.6e-8;
max difference 2.7e-7 R-SE units; SEs equal to 4 decimals; no boundary fits (log SD intercept
between -0.97 and -0.72). See the test header for the cells.
