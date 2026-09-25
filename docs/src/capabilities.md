# Detailed capabilities and limits

Use this page **after** choosing a scientific route, not as a first tutorial.
For a first distributional model, start with [Getting started](getting-started.md).
For a tree, start with [Phylogenetic structured effects](tutorials/phylogenetic-models.md).
For Gaussian meta-analysis with supplied sampling variances, start with
[Mean effects and residual heterogeneity](tutorials/meta-analysis.md).

This is the detailed map of what `DRModels.jl` can fit today. It is deliberately
conservative: “tested” means the route has a worked example and routine checks
for the listed use; “implemented, untested” means it is not yet a recommended
analysis route; and “not available” means the request cannot currently be fit
with DRModels.jl.

Status legend:

- **Tested** — suitable for the listed use after your routine model checks.
- **Impl, untested** — not yet a recommended analysis route.
- **Not available** — this request cannot currently be fit with DRModels.jl.

Use [What can I fit today?](model-guides/model-map.md) for a model-building
route, and [Getting started](getting-started.md) for runnable syntax.

## Response families

Each family below can be fitted with fixed effects. The family-level tests use
simulated data with known parameters; comparisons with the optional R bridge are
run separately.

| Family | Mean-axis random effects | Status and boundary |
|---|---|---|
| Gaussian | intercepts, slopes, correlated, crossed, and structured effects | **Tested** |
| Student-t | intercepts and slopes | **Tested** |
| SkewNormal | — | **Tested** fixed effects; random effects are refused |
| Poisson | intercepts, slopes, crossed, and phylogenetic effects | **Tested** |
| NegBinomial2 | intercepts, slopes, crossed, and phylogenetic effects | **Tested** |
| TruncatedNegBinomial2 | — | **Tested** fixed effects |
| Beta | intercepts, slopes, crossed, and phylogenetic effects | **Tested**; crossed random effects have numerical tests but no complete worked analysis |
| BetaBinomial | intercepts, slopes, crossed, and phylogenetic effects | **Tested**; constant `sigma` only |
| Binomial | intercepts, crossed, and phylogenetic effects | **Tested**; slope random effects are refused |
| Gamma | intercepts, slopes, crossed, and phylogenetic effects | **Tested**; crossed random effects have numerical tests but no complete worked analysis |
| LogNormal | intercepts, slopes, phylogenetic, and known-matrix effects | **Tested**; `animal` and coordinate-spatial effects are refused |
| ZeroOneBeta | — | **Tested** fixed effects |
| Tweedie | intercepts and independent slopes | **Tested** |
| CumulativeLogit (ordinal) | intercepts, independent slopes, and phylogenetic effects | **Tested** |

**Count modifiers** `zi` (zero-inflation) and `hu` (hurdle): implemented in the
Poisson/NB2 paths; **Tested**.

**Beta boundary modifiers** `zoi` / `coi` (zero-/one-inflation): the
`ZeroOneBeta()` family handles the boundary mass; **Tested**.

## Distributional (location–scale) sub-models

A formula per distributional parameter is the core grammar: `bf(...)` gives
the mean, scale, and any additional parameter supported by the chosen family
their own formulas.

| Capability | Status and boundary |
|---|---|
| Gaussian mean μ and scale σ formulas | **Tested**; both intercepts and slopes are recovered. |
| Non-Gaussian `sigma` or dispersion formula | **Tested** for families with a dispersion model. |
| Student-t `nu` (degrees of freedom) formula | **Tested**. |
| Random effect on the Gaussian scale axis, `sigma ~ (1\|g)` | **Tested** with Gauss–Hermite integration. |
| `sigma(fit)` and `corpairs(fit)` | **Tested** post-fit accessors. |

## Location–scale–scale models (LSS, `sd()`)

A third submodel can put a linear predictor on the log standard deviation of a
random effect: `sd(group) ~ z` or `sd(species, phylogenetic) ~ z`.

| Capability | Status and boundary |
|---|---|
| Plain IID LSS `sd(group) ~ z` | **Tested** under ML and opt-in REML. |
| Phylogenetic LSS `sd(species, phylogenetic) ~ z` | **Tested** under ML and opt-in REML; large phylogenetic fits use the sparse engine. |
| Sparse phylogenetic LSS | **Tested.** Request `sparse = true` or `algorithm = :sparse_lbfgs`. |
| Multi-component LSS | **Tested** for multi-IID and IID-plus-phylogenetic structures. |
| Incomplete responses | **Tested** for `missing` or `NaN` outcomes on LSS routes. |


## Random-effect structures

### Plain (unstructured) random effects on the mean

| Structure | Status and boundary |
|---|---|
| Gaussian random intercept `(1\|g)` | **Tested** |
| Gaussian random slope `(x\|g)` | **Tested** |
| Correlated intercept and slope | **Tested** |
| Multiple, crossed, or nested grouping factors | **Tested** |
| Poisson crossed random effects | **Tested** with sparse Laplace integration |

### Structured random effects with a known relatedness matrix (closed-form Gaussian)

A structured intercept `u ~ N(0, σ_s² K)` keeps the Gaussian marginal exactly
Gaussian and is fit in closed form (PGLS / matrix-determinant lemma).

| Marker | Supplies | Status |
|---|---|---|
| `relmat(1\|id)` | user matrix `K` | **Tested** |
| `animal(1\|id)` | additive-relatedness `A` | **Tested** |
| `phylo(1\|species)` on the **mean** | tree (`AugmentedPhy` or Newick) | **Tested** |
| `spatial(1\|site)` | coordinates; `K(ρ)=exp(-d/ρ)`, with ρ estimated | **Tested** |
| One `phylo`/`relmat`/`animal` marker **plus** ordinary `(1\|h)` or `(0 + x\|h)` bars on the mean | tree / `K` / `A` | **Tested**, ML only; matches drmTMB `engine = "tmb"` on nine test datasets. Independent blocks; the marker SD is on the correlation scale. REML, `(1 + x\|h)`, range-estimated `spatial()`, `meta_V()`, `penalty` and sparse algorithms are refused by name. |

The Gaussian table above describes the simple intercept route. It does not rule
out the supported non-Gaussian phylogenetic mean models or the more specific
phylogenetic location--scale routes below; those routes have their own family
and inference boundaries.

### Non-Gaussian phylogenetic random intercept on the mean (sparse Laplace)

A `phylo(1|species)` intercept on the mean for non-Gaussian families uses the
verified sparse augmented-state Laplace engine.

| Family | Status |
|---|---|
| Poisson | **Tested** |
| NegBinomial2 | **Tested** |
| Gamma | **Tested** |
| Binomial | **Tested** |
| Beta | **Tested**; its gradient tolerance is deliberately looser than 1e-6 |
| BetaBinomial | **Tested**; constant `sigma` only |
| CumulativeLogit (ordinal) | **Tested**; intercept-only |

## Location–scale with a phylogenetic random effect on the scale (q=2 route)

A shared random effect on **both** the mean and log-dispersion axes uses a q=2
augmented Laplace calculation and exact O(p) outer gradient. The full fit,
gradient, Wald summaries, and profile-likelihood intervals have regular
package-test coverage.

| Capability | Status |
|---|---|
| Coupled mean and log-dispersion axes for NB2 and Gamma | **Tested** through the public `drm()` interface |
| Beta and Beta-binomial kernels | Not available through the public coupled location–scale interface |
| Poisson and lognormal leaves | **Implemented, untested**; do not rely on them for fitted location–scale models |

!!! note
    For **non-Gaussian** responses, coupled location–scale random effects are
    currently available only for NB2 and Gamma. Other non-Gaussian families can
    place structured or ordinary random effects on the mean axis, not the scale
    axis.

    **Gaussian has a separate, tested scale-phylogeny route.** Write
    `bf(y ~ … + phylo(1|sp), sigma ~ phylo(1|sp))` for the default separate
    mean- and scale-phylogeny effects. A scale-only phylogeny is also supported.
    `phylo_coupled = true` opts into a free mean–scale phylogenetic correlation;
    the plain two-`phylo` syntax does not do this. Retrieve the two standard
    deviations with `gaussian_locscale_phylo_sds(fit)`; a coupled fit records the
    correlation in `fit.scales[:lambda_cor]`, and `profile_ci = true` adds profile
    intervals for the two standard deviations.

    This Gaussian route requires a common grouping factor and one phylogenetic
    structured component; it cannot be combined with additional random effects or
    `meta_V`. `method = :REML` is available for the separate and scale-only
    blocks, but is refused for the coupled block and for the iid
    `sigma ~ (1|g)` route. A non-Gaussian scale-axis-only intercept is not part of
    the public formula grammar.

## Coevolution: q=4 phylogenetic bivariate location–scale model (PLSM)

The selling-point model: a shared phylogenetic random effect on all four axes
`(μ1, μ2, log σ1, log σ2)` with a 4×4 between-species covariance `Σ_a`, plus a
residual correlation ρ12. The sparse-Laplace fit, exact O(p) gradient, and
public formula route are **tested**.

| Capability | Status |
|---|---|
| `bf(mu1=…, mu2=…, sigma1=…, sigma2=…, rho12=…)` with phylogeny | **Tested**; recovers fixed effects and `Σ_a` |
| `relmat`, `animal`, and fixed-range `spatial` structured providers | **Tested**; spatial range is fixed (default: mean pairwise distance) |
| `vc(fit)`, `ranef(fit)`, and `coevolution_cor` summaries | **Tested**; `fit.ranef.Sigma_a` is ordered `mu1, mu2, sigma1, sigma2` |
| Fixed-effect covariance and Wald standard errors (`q4_vcov=true`) | **Tested** |
| Bootstrap intervals for coevolution correlations | **Tested for tree-based phylogeny**; non-tree structured providers return point summaries only |

## Structured q=2 bivariate Gaussian (mu1/mu2 only)

This is a complete-response exact-Gaussian ML model with matching
structured random intercepts on `mu1` and `mu2`. It requires the same fixed-effect
design on both mean formulas and intercept-only `sigma1`, `sigma2`, and `rho12`
formulas. Current support covers point estimates and exported summaries only.
It does not cover q2 REML, calibrated intervals, non-Gaussian q2 models, or the
full R bridge.

| Capability | Status |
|---|---|
| `phylo(1\|species)` on `mu1` and `mu2`, with residual `rho12` | **Tested** through `drm()` and direct export |
| `relmat(1\|id)` and `animal(1\|id)` on `mu1` and `mu2`, using known `K` / `A` | **Tested** |
| Fixed-covariance spatial q2 fit | Available only in the documented fixed-covariance example; the range-estimating `spatial(...)` formula route is rejected |

## Bivariate and paired responses (residual correlation)

The Gaussian, lognormal and Student-t routes here fit two responses jointly with a
residual/scatter `rho12`; `associate_pairs` instead couples two *already-fitted*
univariate models in a second stage. Only the lognormal route reaches the
structured (`phylo`/`relmat`/`animal`/`spatial`) engines of the two sections
above, and it does so by delegation rather than by a second engine.

| Capability | Status |
|---|---|
| Bivariate Gaussian with residual `rho12` (`cbind` / `mu1`,`mu2`) | **Tested** |
| `rho12(fit)` accessor | **Tested** |
| Bivariate **lognormal** (`drm(bf(…), LogNormal())`, drmTMB's `biv_lognormal()`) | **Tested.** Both responses must be strictly positive and are modeled as bivariate normal on `log(Y)`: `mu1`/`mu2` are log-scale means and `rho12` is a log-residual correlation, not the raw-scale Pearson correlation. Structured `phylo` and `relmat` routes are tested; `animal` and `spatial` are implemented but untested on this route. `method = :REML` is refused for all bivariate LogNormal models. |
| Bivariate **Student-t** (`drm(bf(…, nu = …), Student())`, drmTMB's `biv_student()`) | **Tested.** `sigma1`/`sigma2` are scale parameters, not marginal SDs; `rho12` is a scatter correlation; and `nu = 2 + exp(η)`, so `nu > 2`. One `nu` is shared by the two responses (it may vary by row via `nu ~ x`); zero `rho12` does not imply independence at finite `nu`. This is a residual-only model: `phylo`, `relmat`, `animal`, `spatial`, and `method = :REML` are deliberately refused. |
| **Staged pair association** — `associate_pairs` / `latent_normal` / `association` / `PairAssociation` / `integration_diagnostics` (drmTMB's `associate_pairs()`) | **Tested.** This is a two-stage, frozen-margin estimator, not a joint model. It supports `gaussian_bernoulli`, `gaussian_nbinom2`, `bernoulli_bernoulli`, `bernoulli_nbinom2`, and `nbinom2_nbinom2`; integration diagnostics are available where numerical integration is used. Its uncertainty ignores margin-estimation error, and it offers no simultaneous association bands or profile intervals. The kernel must be explicit; only `association ~ 1` is supported; `marginal = :AGHQ`, non-converged margins, non-Bernoulli binomial margins, and other pair classes are refused. |
| Cross-family bivariate (different families on `y1` vs `y2`) | **Experimental.** `drm(bf(...), (Gaussian(), Poisson()); data = …)` uses a latent-scale scalar correlation in `fit.rho_latent`. It has narrow documented evidence and no interval-coverage claim. `rho12 ~ x` is refused: this route fits a scalar latent correlation, not an observation-specific residual correlation. See [Cross-family methods](model-guides/cross-family-methods.md). |

## Meta-analysis

| Capability | Status |
|---|---|
| `gaussian()` + `meta_V(v)` with **known diagonal** sampling variances; τ on the σ intercept | **Tested** |
| Bivariate known sampling covariance (`meta_vcov_bivariate`) | **Tested** |
| Deprecated `meta_known_V` parity stub | — | **Not available**; use `meta_V` instead. |

## Inference

| Method | Status |
|---|---|
| Wald SEs + CIs (observed information) | **Tested** |
| Profile-likelihood CIs (`profile_result`, `confint(:profile)`) | **Tested** |
| Parametric bootstrap (`bootstrap_ci`/`_summary`/`_result`, serial + threaded) | **Tested** |
| REML for fixed-effect Gaussian location–scale and Gaussian mean `(1 \| g)` | **Tested** |
| `reml_loglik` / `ml_loglik` / `estimation_method` accessors | **Tested** |
| Epsilon-method bias correction (`bias_correct`) | **Tested** |
| **χ̄² (chi-bar-square) boundary inference** (Self–Liang / Stram–Lee mixture) | **Tested** |
| REML on q=4 Laplace and Location–Scale–Scale models | **Tested** |

!!! warning "REML scope"
    `method=:REML` is opt-in. **ML is the default** (REML likelihoods are not
    comparable across fixed-effect structures). Supported models are the fixed-effect
    Gaussian location–scale model; a single Gaussian mean intercept `(1 | g)`;
    Location–Scale–Scale models (`sd(g) ~ z`,
    `sd(species, phylogenetic) ~ z`, and multi-component LSS); and the
    bivariate q=4 location–scale engine.
    σ-RE, random slopes, and non-Gaussian REML stay rejected. This is not AI-REML.

    **Normalisation convention:** every REML route
    in DRModels.jl now reports the **normalised** Patterson–Thompson restricted
    log-likelihood, so `reml_loglik` is directly comparable to lme4's,
    glmmTMB's, TMB's and drmTMB's `logLik()`. The bivariate q=2/q=4 Laplace
    routes previously omitted the `(n_β/2)·log(2π)` constant while the
    fixed-effect location–scale and mean `(1 | g)` routes included it. That
    inconsistency has been corrected; all supported REML models now report the
    same normalised quantity.

## Model comparison & accessors

| Capability | Status |
|---|---|
| `lrtest`, `anova`, `aicc`, `weights`, `update` | **Tested** |
| `aic` / `bic` / `dof` / `nobs` / `deviance` / `dof_residual` | **Tested** |
| `coef` / `vcov` / `confint` / `stderror` / `coeftable` | **Tested** |
| `fixef` / `re_sd` / `vc` / `ranef` / `sigma` / `corpairs` | **Tested** |
| `family` accessor | **Tested** |
| `heritability` / `repeatability` / `icc` with delta + profile CIs | **Tested** |
| Drop-in parity accessors (StatsAPI surface) | **Tested** |

## Prediction, post-fit, residuals, simulation

| Capability | Status |
|---|---|
| `fitted` / `residuals` / response-scale `predict` | **Tested** |
| `predict_parameters` / `marginal_parameters` / `prediction_grid` / delta-method prediction SEs | **Tested** |
| `simulate` / `check_drm` / Dunn–Smyth quantile residuals | **Tested** |
| Visualization data (`profile_curve` / `parameter_surface` / `corpairs_data`) | **Tested** |
| Drawing (`drm_figure` / thin `plot_*`; Confidence Eye on `:profile`) | Optional with Makie and AlgebraOfGraphics; package checks cover the no-Makie fallback, not rendered figures |

## R → Julia bridge (engine = "julia")

The optional R bridge for `drmTMB(..., engine = "julia")` converts formulas and
data into a form that Julia can fit, then converts the result back to R.

| Capability | Status |
|---|---|
| `drm_bridge` (string/dict/named-tuple formula → fit → flattened `Dict`); univariate, bivariate, phylo-mean, and narrow q2 structured Gaussian examples | **Tested** for the listed model configurations |
| q2/q4 direct-export helpers (`q2_point_export`, `q4_point_export`) | **Tested** for point summaries only; broader bridge behaviour and interval calibration have not been established |
| `drm_bridge_inference` (profile + bootstrap), limited to the Gaussian phylo SD block (`param=:resd`) | **Tested** |
| Newick tree string parsing + small LRU cache | **Tested** |
| Full R-side glue / `engine="julia"` round-trip in drmTMB | (R repo) | **Not available here** — the Julia primitive is tested; the R package glue lives in the drmTMB repository |

## Marginal method selection (VA/ELBO)

| Capability | Status |
|---|---|
| `marginal=:LA` (Laplace) — the default | **Tested** |
| `marginal=:VA` Poisson `(1\|g)` public path | **Experimental**; it uses an ELBO approximation, labels the fit `:VA`, and refuses mixed LA/VA AIC or likelihood-ratio comparisons |
| `marginal=:VA` Binomial / NB2 / Gamma / Beta `(1\|g)` | **Experimental**; scale families require `sigma ~ 1` |
| `method=:VA` on non-Gaussian `drm()` | **Rejected** — choose `marginal=:VA`; `method` is ML/REML |

## Absent / out-of-scope (explicit)

To avoid overclaiming, note these boundaries:

- **Missing data:** listwise predictor preprocessing is not FIML. Experimental
  joint missing-predictor routes (`mi()`, `JointDrmFit`, `JointTwoDrmFit`,
  `JointFiniteDrmFit`, `imputed`, and `miss_control`) are exported for evaluation,
  but their API and numerics may change and they are not covered by R-parity
  evidence. Gaussian observed-response masking and missing-response handling for
  Location–Scale–Scale `sd()` models are available. General multiple imputation
  outside the joint-model routes and an `na.action`-style option are absent.
- **VA/ELBO:** experimental random-intercept VA is limited to `(1|g)` for
  Poisson, Binomial, NB2, Gamma, and Beta (`sigma ~ 1` where applicable).
  Phylogenetic, crossed, correlated-slope, and zero-inflated/hurdle variants are
  not available.
- **Experimental prototypes:** experimental optimisation and diagnostic prototypes
  are not public analysis methods. The supported REML and `algorithm = :em`
  routes are listed in the Inference table.

## Current evidence boundaries

- `drm_bridge_inference` is tested only for the Gaussian phylogenetic standard
  deviation block (`param=:resd`); other bridge inference parameters are not yet
  supported claims.
- Coupled q=2 location–scale fitting is public and tested for NB2 and Gamma.
  Beta and beta-binomial cannot currently be requested through `drm()` and
  `bf()` for this model. Poisson and LogNormal versions are implemented but have
  not yet been tested as complete analyses.

---

*“Tested” supports the listed use, not a package-wide performance or
interval-coverage claim. Always assess your fitted model and study design.*
