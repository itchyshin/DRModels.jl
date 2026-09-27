# Implementation map

!!! note "Status — Implemented"
    Mirrors drmTMB's [Implementation map](https://itchyshin.github.io/drmTMB/articles/implementation-map.html). Where the **source map** lists *files* and the **model map** lists *capability status*, this page maps each **modelling feature → the numerical method that fits it → the fitter entry point** in the source. All entries are taken from `src/`.

Every fit minimises a negative log-likelihood by L-BFGS. Most routes take their
gradient from ForwardDiff; the verified sparse engines below instead supply an
exact O(p) analytic gradient, because ForwardDiff duals do not flow through the
sparse CHOLMOD factorisation those routes rely on. The rows below differ in
**how the random-effect / correlation integral is handled**.

## The verified engine (phylogenetic q=4 PLSM)

| Feature | Method | Entry point |
|---|---|---|
| q=4 phylogenetic bivariate location–scale | sparse **augmented-state Laplace** + exact **O(p)** gradient (implicit-function, Takahashi selected inverse — never forms dense Σ) | `fit_q4_sparse_tmb` |

## Gaussian location–scale

| Feature | Method | Entry point |
|---|---|---|
| fixed effects | direct ML | `_fit_fixed_gaussian` |
| random intercept `(1\|g)` | closed-form marginal (Woodbury / determinant lemma) | `_fit_ranef_gaussian` |
| correlated `(1+x\|g)` | per-group 2×2 block capacitance, log-Cholesky Σ | `_fit_correlated_ranef_gaussian` |
| crossed `(1\|g)+(1\|h)` | whitened-Woodbury dense capacitance `M = I + Z̃ᵀD⁻¹Z̃` | `_fit_multi_ranef_gaussian` |
| scale RE `sigma ~ (1\|g)` | 32-node Gauss–Hermite marginal on log σ (default); per-group 1-D Laplace with `marginal = :Laplace` | `_fit_sigma_ranef_gaussian` |
| `relmat` / `animal` / `phylo` | structured GLS (determinant lemma + Woodbury, known K) | `_fit_structured_gaussian` |
| `spatial(1\|site)` | exponential kernel `K(ρ)=exp(-d/ρ)`, range estimated | `_fit_spatial_gaussian` |
| meta-analysis `meta_V(v)` | known-variance GLS + heterogeneity τ | `_fit_meta_gaussian` |

## Non-Gaussian families

The shared scheme: an explicit, AD-safe log-likelihood; random intercepts by
Gauss–Hermite quadrature (`b = √2 σ_b z`); correlated `(1+x|g)` by a 2-D GHQ
tensor grid; crossed intercepts by the sparse-Laplace spine
(`sparse_laplace_glmm.jl`).

| Family | Fixed | `(1\|g)` | `(1+x\|g)` | crossed | other |
|---|---|---|---|---|---|
| Student-t | `_fit_student` | `_fit_student_ranef` | `_fit_student_corr_ranef` | — | — |
| Poisson | `_fit_poisson` | `_fit_poisson_ranef` | `_fit_poisson_corr_ranef` | `_fit_poisson_crossed_laplace` | `zi` / `hu`: `_fit_poisson_zi` / `_fit_poisson_hu` |
| NB2 | `_fit_negbin2` | `_fit_negbin2_ranef` | `_fit_negbin2_corr_ranef` | `_fit_nb2_crossed_laplace` | `zi` / `hu`: `_fit_negbin2_zi` / `_fit_negbin2_hu`; truncated: `_fit_truncated_negbin2` |
| Gamma | `_fit_gamma` | `_fit_gamma_ranef` | `_fit_gamma_corr_ranef` | `_fit_gamma_crossed_laplace` | — |
| Beta | `_fit_beta` | `_fit_beta_ranef` | `_fit_beta_corr_ranef` | `_fit_beta_crossed_laplace` | — |
| Binomial | `_fit_binomial` | `_fit_binomial_ranef` | — | `_fit_binomial_crossed_laplace` | — |
| Beta-binomial | `_fit_betabinomial` | `_fit_betabinomial_ranef` | `_fit_betabinomial_corr_ranef` | — | `cbind(s, f)` response |
| LogNormal | `_fit_lognormal` | `_fit_lognormal_ranef` | `_fit_lognormal_corr_ranef` | — | — |
| Zero-one-inflated Beta | `_fit_zeroonebeta` | — | — | — | closed `[0,1]` |
| Tweedie | `_fit_tweedie` | `_fit_tweedie_ranef` | — | — | compound Poisson–Gamma; independent slope `_fit_tweedie_slope_ranef` (correlated `(1+x\|g)` refused) |
| Cumulative-logit | `_fit_cumulative` | `_fit_cumulative_ranef` | — | — | ordered categorical; independent slope `_fit_cumulative_slope_ranef`, phylo `_fit_cumulative_phylo_laplace` (correlated `(1+x\|g)` refused) |

## Shared spine

| Concern | Method | Entry point |
|---|---|---|
| crossed / structured non-Gaussian RE | sparse-Laplace marginal + exact nuisance gradient | `sparse_laplace_glmm.jl` (`_fit_crossed_mean_laplace`, `…_nuisance`) |
| selected inverse for the O(p) correction | Takahashi recursion on the sparse Cholesky | `takahashi_selinv` |
| inference | Wald (observed information), profile (LR inversion), parametric bootstrap | `inference.jl` |

Function names are internal (the public verbs are `drm` / `bf` and the accessors);
this map is for orientation when reading or extending the source.
