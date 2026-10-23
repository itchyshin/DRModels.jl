# Implemented source map

!!! note "Status — Stable"
    Mirrors drmTMB's [Implemented source map](https://itchyshin.github.io/drmTMB/articles/source-map.html). This page is a guided tour of the main `src/` files wired into the module (via `src/DRModels.jl`), taken straight from the source; it is not the complete include list — `src/DRModels.jl` itself is. The `src/experimental/` migrations are **not** yet part of the public API.

DRModels.jl loads in three layers: the **verified q=4 engine**, the **`bf`/`drm` front
end + Gaussian family**, and the **non-Gaussian families** on a shared
sparse-Laplace spine — then post-fit, inference, and output. Everything below is
`include`d from [`src/DRModels.jl`](https://github.com/itchyshin/DRModels.jl/blob/main/src/DRModels.jl).

## Verified engine core

The selling-point model — the q=4 phylogenetic bivariate location–scale model
(PLSM) — fits through a sparse augmented-state Laplace approximation with an
exact O(p) gradient. The files form a single migrated chain (top calls down):

| File | Role |
|---|---|
| `fit_q4_sparse_tmb.jl` | Sparse "TMB-like" exact-gradient fit of the q=4 PLSM — the engine entry point (`include`s the chain below). |
| `fit_ml_q4.jl` | ML fit by ascending the true ML Laplace objective. |
| `sparse_em_fit.jl` | Laplace-EM on the validated sparse augmented foundation. |
| `sparse_aug_plsm.jl` | The sparse augmented-state Laplace-EM core for the q=4 PLSM. |
| `sparse_phy.jl` | Augmented-state sparse phylogenetic precision (never forms a dense p×p Σ). |
| `takahashi_selinv.jl` | Takahashi selected inverse for sparse positive-definite matrices (the O(p) gradient correction). |
| `reml_q4.jl` | REML for the q=4 PLSM (`drm(method = :REML)`): the restricted objective over the augmented state `(u, β)` with its exact O(p) gradient — see [The exact gradient of the q=4 REML objective](reml-q4-exact-gradient.md). |

## Front end + Gaussian family

| File | Role |
|---|---|
| `gaussian_core.jl` | Public formula front end (`bf` / `drm_formula` / `drm`) + the univariate Gaussian location–scale fitter. |
| `gaussian_bivariate.jl` | Bivariate Gaussian location–scale with predictor-dependent residual correlation `ρ12` (keyword `bf`). |
| `gaussian_ranef.jl` | Ordinary Gaussian random intercepts / slopes on the mean (and the scale-RE GHQ marginal). |
| `gaussian_meta.jl` | Gaussian meta-analysis with known sampling (co)variances (`meta_V`). |
| `gaussian_structured.jl` | Structured random effects (`relmat` / `animal` / `phylo` / `spatial`) on the Gaussian mean. |
| `location_only.jl` | Exact-Gaussian location-only phylogenetic mean REML diagnostics and internal row contracts. |

## Non-Gaussian families

All non-Gaussian families share one reusable Laplace spine:

| File | Role |
|---|---|
| `sparse_laplace_glmm.jl` | Reusable sparse-Laplace spine for non-Gaussian GLMMs (the crossed/structured RE path). |
| `student.jl` | Student-t: robust location–scale–shape regression. |
| `poisson.jl` | Poisson: log-link counts. |
| `negbinomial.jl` | Negative-binomial (NB2) for overdispersed counts (incl. the truncated variant). |
| `beta.jl` | Beta for responses on the open interval (0, 1). |
| `betabinomial.jl` | Beta-binomial: successes out of known trials, overdispersed. |
| `binomial.jl` | Binomial / Bernoulli: classic logistic regression. |
| `gamma.jl` | Gamma for strictly-positive continuous responses. |
| `lognormal.jl` | LogNormal for strictly-positive responses whose log is Gaussian. |
| `zeroonebeta.jl` | Zero-one-inflated beta for the closed interval [0, 1]. |
| `tweedie.jl` | Tweedie (compound Poisson–Gamma, 1 < p < 2): semicontinuous responses. |
| `cumulative.jl` | Cumulative-logit ordinal regression. |

## Post-fit, inference, and output

| File | Role |
|---|---|
| `inference.jl` | Wald + profile-likelihood inference (and parametric bootstrap) for a fitted `DrmFit`. |
| `summary.jl` | Human-readable printout for a fitted `DrmFit`. |
| `visualization.jl` | Plotting-*data* providers (`profile_curve` / `parameter_surface` / `corpairs_data`), mirroring drmTMB's visualization layer. Drawing itself is an optional extension — `src/plotting_ext.jl` + `ext/DRModelsMakieExt.jl` — whose methods load only with the Makie + AlgebraOfGraphics weakdeps. |

## Not yet wired — `src/experimental/`

Some migrated comparison-suite engines and natural-gradient variants remain
outside the public API. Several are kept as **recorded negative results**, not
as pending promotions: `fit_em_natgrad.jl` did not meet the promotion criteria,
so `algorithm = :natgrad` is deliberately not exposed, and `fit_em_closed.jl` /
`em_squarem_fit.jl` rest on a closed-form Λ step whose apparent descent was an
artefact of a dropped-zeros sparsity pattern. `test/test_lambda_p100.jl` now
asserts that the step *ascends* the marginal, so these alternatives stay
unwired unless there is a new technical case for promotion. The rest are
superseded predecessors of the production engine (the `estep_*` mode-finder
variants) or diagnostic oracles kept for reproducibility. See
`src/experimental/README.md` for the current boundary.
