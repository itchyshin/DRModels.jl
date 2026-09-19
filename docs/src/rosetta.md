# Rosetta — R ↔ Julia

DRModels.jl is the Julia twin of [drmTMB](https://itchyshin.github.io/drmTMB/), so the
modelling grammar is intentionally parallel: the same `bf()` formula bundle, the
same distributional-parameter names, the same structured-effect markers. This
page is a side-by-side phrasebook for translating a drmTMB (R) call into DRModels.jl
(Julia).

Three differences cover almost everything:

| | drmTMB (R) | DRModels.jl (Julia) |
|---|---|---|
| **Fit verb** | `drmTMB(bf(...), family = ...)` | `drm(bf(...), Family(); data = ...)` |
| **Family** | lower-case function — `gaussian()` | capitalised struct — `Gaussian()` |
| **Scale parameter** | `sigma` | `sigma` (never `tau`) |

!!! note "On the R column"
    The R snippets show drmTMB's grammar (which itself mirrors **brms**). The
    family-constructor and S3 method spellings here were reconciled (2026-06-03)
    against drmTMB's exported interface; the
    parameterisations (e.g. Beta `φ = 1/σ²`) match. drmTMB reuses the base-R
    `stats` families (`gaussian()`, `poisson()`, `Gamma()`, `binomial()`) rather
    than redefining them. This page is maintained from the Julia side.

## The fit call

```r
# R — drmTMB
fit <- drmTMB(bf(y ~ x, sigma ~ x), family = gaussian(), data = dat)
```

```julia
# Julia — DRModels.jl
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = dat)
```

`bf()` keeps the same shape in both: the first formula's left-hand side is the
response and its `μ` predictor; each later `param ~ …` formula sets that
distributional parameter (`sigma` defaults to `~ 1`).

## Families

| drmTMB (R) | DRModels.jl (Julia) | extra parameters |
|---|---|---|
| `gaussian()` | `Gaussian()` | `sigma` |
| `student()` | `Student()` | `sigma`, `nu` |
| `skew_normal()` | `SkewNormal()` | `sigma`, `nu` (slant `alpha`; moment form on both sides) |
| `poisson()` | `Poisson()` | — (mean only; `zi`, `hu`) |
| `nbinom2()` | `NegBinomial2()` | `sigma` (dispersion `θ = 1/σ²`); `zi`, `hu` |
| `truncated_nbinom2()` | `TruncatedNegBinomial2()` | `sigma` (dispersion `θ = 1/σ²`) |
| `beta()` | `Beta()` | `sigma` (precision `φ = 1/σ²`) |
| `beta_binomial()` | `BetaBinomial()` | `sigma` (`φ = 1/σ²`) |
| `binomial()` | `Binomial()` | — (mean only) |
| `Gamma()` | `Gamma()` | `sigma` (CV; shape `α = 1/σ²`) |
| `lognormal()` | `LogNormal()` | `sigma` |
| `zero_one_beta()` | `ZeroOneBeta()` | `sigma`, `zoi`, `coi` |
| `tweedie()` | `Tweedie()` | `sigma` (`φ`), `nu` (power `p`) |
| `cumulative_logit()` | `CumulativeLogit()` | — (ordered cutpoints) |
| `biv_gaussian()` | `Gaussian()` + `bf(mu1=…, mu2=…, rho12=…)` | `sigma1`, `sigma2`, `rho12` |

## Formula grammar

| Intent | drmTMB (R) | DRModels.jl (Julia) |
|---|---|---|
| Mean + scale | `bf(y ~ x, sigma ~ x)` | `bf(@formula(y ~ x), @formula(sigma ~ x))` |
| Extra parameter | `bf(y ~ x, sigma ~ 1, nu ~ 1)` | `bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))` |
| Random intercept | `y ~ x + (1 \| g)` | `@formula(y ~ x + (1 \| g))` |
| Random slope | `y ~ x + (1 + x \| g)` | `@formula(y ~ x + (1 + x \| g))` |
| Crossed REs | `y ~ x + (1 \| g) + (1 \| h)` | `@formula(y ~ x + (1 \| g) + (1 \| h))` |
| Two-column response | `cbind(s, f) ~ x` | `@formula(cbind(s, f) ~ x)` |
| Zero-inflation | `bf(y ~ x, zi ~ 1)` | `bf(@formula(y ~ x), @formula(zi ~ 1))` |
| Hurdle | `bf(y ~ x, hu ~ 1)` | `bf(@formula(y ~ x), @formula(hu ~ 1))` |

### Bivariate (two responses + residual correlation)

```r
# R — drmTMB
drmTMB(bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~ x, sigma2 = ~ 1, rho12 = ~ 1),
       family = biv_gaussian(), data = dat)
```

```julia
# Julia — DRModels.jl  (keyword form; ρ12 is the residual correlation, on atanh ρ12)
bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
   sigma1 = @formula(sigma1 ~ x), sigma2 = @formula(sigma2 ~ 1),
   rho12 = @formula(rho12 ~ 1))
```

### Structured effects & meta-analysis

| Intent | drmTMB (R) | DRModels.jl (Julia) |
|---|---|---|
| Relatedness matrix | `y ~ x + relmat(1 \| id)`, `K = K` | `@formula(y ~ x + relmat(1 \| id))`, `K = K` |
| Animal model | `y ~ x + animal(1 \| id)`, `A = A` | `@formula(y ~ x + animal(1 \| id))`, `A = A` |
| Phylogenetic | `y ~ x + phylo(1 \| species)`, `tree = tree` | `@formula(y ~ x + phylo(1 \| species))`, `tree = tree` |
| Spatial | `y ~ x + spatial(1 \| site)`, `coords = xy` | `@formula(y ~ x + spatial(1 \| site))`, `coords = xy` |
| Meta-analysis (univariate) | `gaussian()` + `meta_V(v)` | `Gaussian()` + `meta_V(v)` |
| Meta-analysis (bivariate, known sampling covariance) | `meta_V(V = meta_vcov_bivariate(...))` **inside** `mu1` | `meta_vcov_bivariate(...)` passed as `drm(...; V = V)` — see [meta-analysis](model-guides/meta-analysis.md) |

## Post-fit accessors

| drmTMB (R) | DRModels.jl (Julia) |
|---|---|
| `coef(fit)` / `fixef(fit)` | `coef(fit)` / `fixef(fit)` |
| `vcov(fit)` | `vcov(fit)` |
| `confint(fit, method = "wald")` | `confint(fit; method = :wald)` |
| `confint(fit, method = "profile")` / `profile(fit)` | `confint(fit; method = :profile)` |
| parametric bootstrap CIs | `bootstrap_ci(bf(...), Family(); data, B = 300)` |
| `logLik(fit)` | `loglik(fit)` |
| `AIC(fit)` / `BIC(fit)` | `aic(fit)` / `bic(fit)` |
| `nobs(fit)` | `nobs(fit)` |
| `deviance(fit)` | `deviance(fit)` |
| `df.residual(fit)` | `dof_residual(fit)` |
| `ranef(fit)` | `ranef(fit)` |
| random-effect SDs | `re_sd(fit)` / `vc(fit)` |
| `sigma(fit)` | `sigma(fit)` |
| `rho12(fit)` | planned (parity gap) |
| `corpairs(fit)` | `corpairs(fit)` / `corpairs_data(fit)` |
| `fitted(fit)` / `residuals(fit)` | `fitted(fit)` / `residuals(fit)` |
| `predict(fit, newdata)` (response mean) | `predict(fit, newdata; type = :response)` |
| predict **every** distributional parameter at new data | `predict_parameters(fit, newdata; type = :response)` |
| in-sample fitted per-observation parameters | `marginal_parameters(fit)` |
| build a covariate grid for prediction | `prediction_grid(reference; predictor = values, …)` |
| `simulate(fit)` | `simulate(fit)` |
| `summary(fit)` | `show(fit)` / `coeftable(fit)` (no `summary` method) |
| `weights(fit)` | planned (parity gap) |
| `family(fit)` | `family(fit)` |
| `is_converged(fit)` / convergence diagnostics | `is_converged(fit)` / `check_drm(fit)` |

### Prediction

drmTMB centres prediction on `predict(fit, newdata)`, which returns the
response-scale mean. DRModels.jl matches that and adds first-class verbs for the
*other* distributional parameters:

| Intent | drmTMB (R) | DRModels.jl (Julia) |
|---|---|---|
| Response-scale mean at new data | `predict(fit, newdata)` | `predict(fit, newdata; type = :response)` |
| Linear-predictor (link) scale | `predict(fit, newdata, type = "link")` | `predict(fit, newdata; type = :link)` |
| Predict **all** distributional parameters | (capability: predict each `bf()` parameter) | `predict_parameters(fit, newdata; type = :response)` |
| In-sample fitted per-obs parameters | (capability: per-observation fitted parameters) | `marginal_parameters(fit)` |
| Build a covariate grid | (capability: hold-others / sweep grid) | `prediction_grid(reference; predictor = values, …)` |

- `predict(fit, newdata; type = :response)` returns the response-scale mean (the
  family inverse link applied to `Xβ̂` — `exp`/`logistic`/identity); `type =
  :link` returns `Xβ̂`. Univariate gives a vector; bivariate gives
  `Dict(:mu1 => …, :mu2 => …)`.
- `predict_parameters(fit, newdata)` returns a
  `Dict(:mu => …, :sigma => …, …)` with one entry per distributional parameter
  the family carries (plus any of `:nu`, `:zi`, `:hu`, `:zoi`, `:coi`); a
  bivariate fit returns `:mu1, :mu2, :sigma1, :sigma2, :rho12`. `type = :response`
  (default) applies each parameter's inverse link (σ via `exp`, ρ12 via `tanh`);
  `type = :link` returns each working-scale linear predictor.
- `marginal_parameters(fit)` reads the stored in-sample fitted parameters
  straight from the fit (no recomputation); in-sample it equals
  `predict_parameters(fit, data)` on the response scale.
- `prediction_grid(reference; predictor = values, …)` builds a `newdata` column
  table by sweeping the named predictors over their value ranges (their Cartesian
  product) while holding every other predictor at its reference (mean for numeric
  columns). It is pure data, so it composes directly:
  `predict_parameters(fit, prediction_grid(...))`.

drmTMB's exact R spelling for per-parameter prediction is not asserted here
(only the capability is); the response-mean `predict` row is the verified parity
point.

## Naming rules to remember

- **Scale is `sigma`, never `tau`.** `bf(y ~ x, tau ~ x)` is rejected — use
  `sigma`. (Group-level phylo/spatial/study variances are reported as named
  covariance summaries, not as a residual scale.)
- **`rho12` is the bivariate *residual* correlation** between the two responses
  (on the `atanh` scale). It is only valid in the keyword `bf(mu1 = …, mu2 = …,
  rho12 = …)` form — not as a univariate parameter.
- **ML is the default.** REML is an option (the likelihoods are not comparable
  across different fixed-effect structures, so ML is used for model selection).

For the experimental, optional R-to-Julia route, see the
[R ↔ Julia bridge](r-julia-bridge.md). It is limited to the documented
admitted models; ordinary `drmTMB` use continues to use the default R engine.
