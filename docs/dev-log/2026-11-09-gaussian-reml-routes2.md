# Gaussian REML, second campaign: scout, pre-registered thresholds, verdicts (2026-11-09)

Reader: whoever owns R-Julia parity of REML. Purpose: say which of the Gaussian REML cells that the wave-10 gap map lists as "Julia behind" this branch closes, on what evidence, and which stay refused and why. Base: `0bfcde2bb` (wave 10, PR #1272); follows `docs/dev-log/2026-11-09-gaussian-reml-routes.md` (wave 9), whose formula and agreement rule it reuses.

## Cells (wave-10 gap map, `maps/julia_behind_ranked.tsv`, Gaussian REML rows)

| Gap-map rank | Cells | Shape | Julia refusal today | Plan |
| --- | --- | --- | --- | --- |
| 1 | 8 | ordinary q = 3, 4 `(1 + x1 + x2 [+ x3] \| g)`, plain and labelled; crossed q = 3, 4 plain and labelled (`+ (1 \| g2)`) | `_re_kind` stops at q = 2 on the REML gate (error text "fitted by maximum likelihood only") | admit |
| 19 | 2 | crossed q = 2 `(1 + x1 \| g) + (1 \| g2)`, plain and labelled | "correlated block beside another term ... no restricted objective" | admit |
| 3 | 4 | relmat, animal q = 1 with `sd(g, level) ~ 1` and `~ z` | "structured intercept SD regression supports maximum likelihood only" | admit |
| 3 and 39 | 2 | spatial q = 1 with `sd(g, spatial) ~ 1` and `~ z` | same | admit (its kernel is already the fixed native one, see below) |
| 5, 9 | 3 | animal, relmat, spatial q = 2 bare `(1 + x1 \| g)`, `sd-none` | `_default_structured_q2_sd` skips REML; parser refuses the term | admit for Gaussian only |
| 39 | 1 | spatial q = 1 bare `spatial(1 \| g)` | REML refused with "ML differs by 0.9" | stays refused: see Rank 39 |

Not attempted: labelled structured REML (R refuses the labelled non-phylo block too), labelled `cm1` (sigma-side random effect, rank 41 to 42), and all non-Gaussian REML.

## What R does (convention per cell)

drmTMB 0.8.1 `REML = TRUE` moves `beta_mu` into TMB's random block with a flat prior. For all cells above `sigma ~ 1` has no variance component, so `beta_sigma` and every variance parameter stay outer. The Laplace step over `(u, beta_mu)` is exact because the model is Gaussian-linear in them, so each cell is the Patterson-Thompson objective

`nll_REML(theta) = nll_ML(theta) + 0.5 * logdet(X' V(theta)^-1 X) - 0.5 * p_mu * log(2 pi)`

(the convention wave 9 verified, and the one the harness R logLik reproduces below to 1e-10). The structured `sd(g, level) ~ z` cells are the same objective with `sd_g = exp(Z alpha)` inside `V`. Cells with a `sigma` predictor, a `sigma` random effect, `meta_V`, or missing responses are not covered by any reference and stay refused.

R itself refuses REML with an `sd()` formula on the ordinary and crossed correlated blocks (every `sd-one`, `sd-z`, `sd-first` REML cell of those shapes is "both refuse" on the gap map), so those stay refused here with a message saying so.

## Rank 39: the ML gap is a kernel difference, not a defect

`spatial(1 | g)` in Julia (`_fit_spatial_gaussian`) estimates the exponential range; drmTMB fixes it at the median positive pairwise distance (`docs/dev-log/2026-11-08-spatial-kernel-audit.md`, row 1: "different model by design, left alone"). Evidence from the gap map on the same data: R ML logLik -393.1792 for both `sd-none` and `sd-one`; Julia's fixed-range route (`sd-one`) reproduces it to 1e-10, while Julia's estimated-range route gives -392.2383 (higher, because it nests R's model). So the 0.94 is the extra range parameter, not an optimiser or likelihood error. Making `spatial(1 | g)` use the fixed range would change the ML fit of an existing documented route, which is not bounded, so the bare REML cell stays refused. The refusal now says why and points to the working spelling: `sd(g, spatial) ~ 1` (fixed native kernel, REML supported, agrees with R's bare-spatial REML logLik).

## Pre-registered agreement rule (fixed before any Julia number was compared with the new R reference)

Per shape, three simulated datasets (seeds 21, 22, 23; n = 200, 40 groups), R drmTMB 0.8.1 `REML = TRUE` on Totoro against Julia `drm(...; method = :REML)`:

1. REML logLik: absolute difference at most 1e-4.
2. Mean coefficients: each `|diff| / SE_R` at most 0.02.
3. Standard deviations on the natural scale (block shapes; the residual SD and each random-effect SD): relative difference at most 1e-2. Structured shapes compare the `sd()` coefficients through `exp(coef)`.
4. Correlations of a block: absolute difference at most 1e-2.
5. SDs below 1e-4 times the response SD on either side are boundary cells: flagged and excluded from rules 3 and 4, not from rule 1.
6. A dataset where R itself does not converge (`opt$convergence != 0` or non-PD Hessian) is excluded from the rules and recorded; a shape that misses a rule on any other dataset is refused, not shipped.
7. Medians and maxima per group are reported.

Verdict categories: (a) fits and agrees, (b) R does not converge (recorded, left refused or excluded), (c) too large.

## Data and reference

`test/fixtures/gaussian_reml_routes2/` (`gen_ref.R` regenerates it on Totoro from the refusal-fuzz covariates; `compare.jl` computes the agreement table). 36 R fits, 35 converged. R does not converge on `ord4`, seed 22 (convergence code 1, non-PD Hessian): excluded by rule 6.
