# Gaussian REML on the generic univariate routes: scout and design (2026-11-09)

Reader: whoever owns R-Julia parity of REML. Purpose: say what R drmTMB 0.8.1 does for the 15 refusal-fuzz cells where R fits with `REML = TRUE` and Julia (wave-8 tip `479bbf963`) refuses with "method = :REML is not implemented for this model on the generic univariate Gaussian route", and which of them this branch closes.

Source of the 15 cells: drmTMB branch `claude/gap-map-wave8-20261108`, `docs/dev-log/parity-crosswalk-evidence-20261107/fuzz-gap-map-wave8/maps/w8/julia_behind_groups.tsv`, rank 1. Every cell has `sigma ~ 1` and mean-side random effects only.

## What R does (read from `R/drmTMB.R`, `drm_apply_estimator_spec`)

`REML = TRUE` moves `beta_mu` into TMB's random block with a flat prior. The model is Gaussian-linear in `(u, beta_mu)`, so the Laplace step is exact and the objective is the Patterson-Thompson restricted likelihood:

`nll_REML(theta) = nll_ML(theta) + 0.5 * logdet(X' V(theta)^-1 X) - 0.5 * p_mu * log(2 pi)`

`beta_sigma` stays an outer parameter unless a sigma variance component exists (none of these 15 cells has one). So all 15 cells are exact (not Laplace-flagged); contrast the structured `sd(g, sigma)` routes, which are `estimator_exact = FALSE`. Julia already uses this same formula and constant on `(1 | g)`, `(1 + x | g)` and the phylo mean-only cell, so a Julia REML logLik is on R's scale.

## The 15 cells, in five groups

| Group | Cells | Julia ML today | R REML | Reusable Julia piece | Recommendation |
| --- | --- | --- | --- | --- | --- |
| A. slope only `(0 + x | g)` | slope q1 (1) | agrees with R to 0 | `_fit_ranef_gaussian` already has `nll_reml(θ)` written for a general weight vector `w`; only the gate restricts it to intercepts | implement: lift the `kind === :intercept` restriction |
| B. several independent scalar terms | crossed q1 `(1\|g)+(1\|g2)`, indep q2, q3, q4 `(1\|g)+(0+x1\|g)+...` (4) | `_fit_multi_ranef_gaussian`, agrees with R | exact, as above | whitened design `Z~ = Z diag(sigma_k)` already built there | implement: restriction term from the same stacked QR |
| C. dense structured intercept `animal` / `relmat` (q1) | animal q1, relmat q1 (2) | `_fit_structured_gaussian`, agrees with R | exact | `Z~ = sigma_s Z L_K` fits the same stacked QR | implement |
| D. phylo intercept plus slope, two SDs `phylo(1 + x \| g)` | phylo q2 (1) | `_fit_phylo_slope_gaussian`, agrees with R | exact | dense V; `Z~ = [sigma_a Z L_C, sigma_b Zx L_C]` | implement |
| E. spatial q1 | spatial q1 (1) | fits, but its ML logLik differs from R's ML by 0.94 (Julia higher) on the committed fuzz data | exact in R | none until the ML gap is explained | refuse with accurate wording; REML cannot agree where ML does not |
| F. crossed with correlated block `(1 + x.. \| g) + (1 \| g2)` | crossed q2, q3, q4, each lab0 and lab1 (6) | Julia ML **refuses** ("a correlated `(1 + x \| g)` block must be the only random-effect term") | exact in R | none: Julia has no correlated block alongside another term | refuse with the ML-level reason; a REML-only admission would be a different feature |

Cell count: 1 + 4 + 2 + 1 + 1 + 6 = 15.

## The shared restriction term

For one stacked least-squares problem `A = [D^-1/2 Z~ ; I]`, `Xs = [D^-1/2 X ; 0]`, the residual matrix `E = Xs - A (A \ Xs)` satisfies `E'E = X' V^-1 X` exactly. It is positive semidefinite by construction (no Woodbury subtraction), the same reasoning Julia's ML routes use to avoid cancellation as `sigma_e -> 0` (#835/#837). One helper `_reml_xtvinvx_qr(Z~, sdinv, X)` serves groups B, C and D. Group A keeps its own existing `_re_xtvinvx_stable`.

## Pre-registered agreement rule (fixed before any R reference was computed)

Per group, at least 3 datasets, R drmTMB 0.8.1 `REML = TRUE` (Totoro library, `engine = "tmb"`) against Julia `drm(...; method = :REML)`:

1. REML logLik: absolute difference at most 1e-4.
2. Mean coefficients: each `|Δ| / SE_R` at most 0.02.
3. Standard deviations on the natural scale: relative difference at most 1e-2, except where either side is on the boundary (SD below 1e-4 times the response SD); boundary cells are flagged and excluded from the SD rule, not from the logLik rule.
4. Reported: median and maximum of each, per group. A group that misses 1 to 3 on any dataset is refused, not shipped.

## What is NOT covered

- Heteroscedastic `sigma` (pσ > 1) on these routes: the formula is still exact, but nothing here tests it, so it is refused under REML.
- Missing responses, `meta_V`, a sigma random effect, penalties: unchanged refusals.
- Groups E and F stay refused.

## Result (2026-11-09)

Groups A to D implemented; E and F refused. 24 fits (3 datasets x 8 designs, n = 200), R drmTMB 0.8.1 REML = TRUE as reference, all converged on both sides, no boundary cells. Maximum differences: logLik 4.7e-12, mean coefficients 4.6e-8 R-SE, SD relative 2.8e-10; every pre-registered threshold met by at least 7 orders of magnitude. Per-fit table: `docs/dev-log/evidence/2026-11-09-gaussian-reml-routes-agreement.tsv`; test: `test/test_gaussian_reml_routes.jl`; data and R reference: `test/fixtures/gaussian_reml_routes/` (`gen_ref.R` regenerates them).
