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
    parameterisations (e.g. Beta `φ = 1/σ²`) match, except where a row below says otherwise (the spatial range). drmTMB reuses the base-R
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
| `poisson()` | `Poisson()` | — (mean only); `zi` on both sides; `hu` (hurdle Poisson) in Julia only |
| `nbinom2()` | `NegBinomial2()` | `sigma` (dispersion `θ = 1/σ²`); `zi` on both sides; Julia also accepts `hu` here (the R hurdle is the next row) |
| `truncated_nbinom2()` | `TruncatedNegBinomial2()` | `sigma` (dispersion `θ = 1/σ²`); add `hu` for the hurdle NB2 model on both sides |
| `beta_family()` | `Beta()` | `sigma` (precision `φ = 1/σ²`) |
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
| Hurdle (NB2) | `bf(y ~ x, sigma ~ 1, hu ~ 1)` with `truncated_nbinom2()` | `bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(hu ~ 1))` with `TruncatedNegBinomial2()` or `NegBinomial2()` |

### Bivariate (two responses + residual correlation)

```r
# R — drmTMB
drmTMB(bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~ x, sigma2 = ~ 1, rho12 = ~ 1),
       family = biv_gaussian(), data = dat)
```

```julia
# Julia — DRModels.jl  (ρ12 is the residual correlation, on atanh ρ12)
fit = drm(
    bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
       sigma1 = @formula(sigma1 ~ x), sigma2 = @formula(sigma2 ~ 1),
       rho12 = @formula(rho12 ~ 1)),
    Gaussian(); data = dat)
```

### Structured effects & meta-analysis

| Intent | drmTMB (R) | DRModels.jl (Julia) |
|---|---|---|
| Relatedness matrix | `y ~ x + relmat(1 \| id)`, `K = K` | `@formula(y ~ x + relmat(1 \| id))`, `K = K` |
| Animal model | `y ~ x + animal(1 \| id)`, `A = A` | `@formula(y ~ x + animal(1 \| id))`, `A = A` |
| Phylogenetic | `y ~ x + phylo(1 \| species)`, `tree = tree` | `@formula(y ~ x + phylo(1 \| species))`, `tree = tree` |
| Spatial | `y ~ x + spatial(1 \| site)`, `coords = xy` (range fixed at the median pairwise distance) | `@formula(y ~ x + spatial(1 \| site))`, `coords = xy` (range fixed at the median pairwise distance, as drmTMB, on every route) |
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
| `rho12(fit)` | `rho12(fit)` |
| `corpairs(fit)` | `corpairs(fit)` / `corpairs_data(fit)` |
| `fitted(fit)` / `residuals(fit)` | `fitted(fit)` / `residuals(fit)` |
| `predict(fit, newdata)` (response scale, random effects at zero) | `predict(fit, newdata; type = :response)` |
| predict **every** distributional parameter at new data | `predict_parameters(fit, newdata; type = :response)` |
| in-sample fitted per-observation parameters | `marginal_parameters(fit)` |
| build a covariate grid for prediction | `prediction_grid(reference; predictor = values, …)` |
| `simulate(fit)` (fresh random effects, `re.form = NA`) | `simulate(fit; data)` (fresh random effects, `re = :marginal`; `re = :zero` sets them to zero; random-effect fits need `data`, except temporal fits can simulate their full marginal chain without it) |
| `summary(fit)` | `summary(fit)` (prints the Wald coefficient table) |
| `weights(fit)` | `weights(fit)`: the prior weights from `drm(...; weights = w)`, or `nothing` (fixed-effect ML routes, Gaussian or Poisson with one `(1 | g)`, Gaussian `meta_V`, and the residual bivariate Gaussian; other routes refuse `weights =`) |
| `drm_control(maxit =, start =, multi_start =, se =)` | `control = DRMControl(maxit =, start =, multi_start =, se =)`; `maxit` / `start` / `multi_start` on fixed-effect routes, `se = false` on every route (`control_info(fit)` reports what it did) |
| `obj$gr(opt$par)` / `sdreport()$cov.fixed` | `gradient(fit)` / `hessian(fit)` (negative log-likelihood, `coef(fit)` order and scale) |
| `family(fit)` | `family(fit)` |
| `is_converged(fit)` / convergence diagnostics | `is_converged(fit)` / `check_drm(fit)` |
| `convergence_status(fit)` (`"converged"` / `"boundary"` / `"degenerate"`) | `convergence_status(fit)` (`:converged` / `:boundary` / `:degenerate`) |

### Prediction

drmTMB centres prediction on `predict(fit, newdata)`, which returns the
response-scale prediction with any random effects set to zero (an average
group). DRModels.jl's `predict` also sets them to zero, so for the GLM-type
families (Gaussian, Poisson, NegBinomial2, Gamma, Beta, Binomial) the two
predictions compare directly; it adds first-class verbs for the *other*
distributional parameters:

| Intent | drmTMB (R) | DRModels.jl (Julia) |
|---|---|---|
| Response-scale prediction at new data (random effects at zero) | `predict(fit, newdata)` | `predict(fit, newdata; type = :response)` |
| Linear-predictor (link) scale | `predict(fit, newdata, type = "link")` | `predict(fit, newdata; type = :link)` |
| Predict **all** distributional parameters | (capability: predict each `bf()` parameter) | `predict_parameters(fit, newdata; type = :response)` |
| In-sample fitted per-obs parameters | (capability: per-observation fitted parameters) | `marginal_parameters(fit)` |
| Build a covariate grid | (capability: hold-others / sweep grid) | `prediction_grid(reference; predictor = values, …)` |

- `predict(fit, newdata; type = :response)` returns `invlink(Xβ̂)` with
  random effects set to 0 (for the GLM-type families, `exp`/`logistic`/identity
  applied to `Xβ̂`; for ZeroOneBeta, `logistic(Xβ̂)`, the Beta-component mean
  rather than the mixture mean; for `LogNormal`, the log-scale `μ`, as in
  drmTMB); `type = :link` returns `Xβ̂`; `type = :mean` returns `E[Y]` at the new
  rows (drmTMB's `type = "mean"`), and `type = :median` is the LogNormal
  `exp(Xβ̂)`. Univariate
  gives a vector; bivariate gives `Dict(:mu1 => …, :mu2 => …)`. With a
  nonlinear link this is the prediction for an average group, not the mean
  averaged over groups.
- `predict_parameters(fit, newdata)` returns a
  `Dict(:mu => …, :sigma => …, …)` with one entry per distributional parameter
  the family carries (plus any of `:nu`, `:zi`, `:hu`, `:zoi`, `:coi`); a
  bivariate fit returns `:mu1, :mu2, :sigma1, :sigma2, :rho12`. `type = :response`
  (default) applies each parameter's inverse link (σ via `exp`, ρ12 via `tanh`);
  `type = :link` returns each working-scale linear predictor.
- `marginal_parameters(fit)` reads the stored in-sample fitted parameters
  straight from the fit (no recomputation); in-sample it equals
  `predict_parameters(fit, data)` on the response scale for the mean and `sigma`
  of the GLM-type families (the `predict_parameters` docstring lists the
  exceptions).
- `prediction_grid(reference; predictor = values, …)` builds a `newdata` column
  table by sweeping the named predictors over their value ranges (their Cartesian
  product) while holding every other predictor at its reference (mean for numeric
  columns). It is pure data, so it composes directly:
  `predict_parameters(fit, prediction_grid(...))`.
- **Fitted values agree by default, with one caveat.** drmTMB's `fitted(fit)` adds the fitted
  random-effect modes to each fitted row and returns the rows used in the fit.
  DRModels.jl's `fitted(fit)` does the same: the default is `re = :conditional`,
  and a fit with a missing response returns the used rows only. The values agree
  with drmTMB's to 1e-6 or better on the Gaussian, LogNormal and phylo routes. On the routes
  that integrate the random intercept by adaptive quadrature (the default for one
  `(1 | g)` on Poisson, NegBinomial2, Binomial, BetaBinomial, Gamma, Beta, Tweedie and
  CumulativeLogit) they differ by about 1e-4 to 5e-3 on the cells checked (Student agreed
  to 1e-9 on its one cell, where the group SD is at the boundary). Here the fitted modes come from the more
  accurate integration (adaptive quadrature), while drmTMB maximises a Laplace
  approximation, so the two packages stop at slightly different parameters. The
  drmTMB `fitted()` is the Laplace value, not an error on either side. A fit made with
  `marginal = :Laplace` on one ordinary `(1 | g)` of Poisson, NegBinomial2, Binomial, Gamma
  or Beta keeps the inner-mode solve at the fitted parameters, so `fitted(fit)` is conditional
  there and agrees with drmTMB's to 1e-6 on the cells checked (the Laplace fit is the same
  objective as drmTMB's). The same holds for a Binomial `(1 + x | g)` block with a
  `:probit` or `:cloglog` link and for a `ZeroOneBeta` `(1 | g)` (here the modes move
  only the Beta component of the mixture mean). Pass
  `fitted(fit; re = :zero)` for the population-level value with every random
  effect at zero, which is what `predict(fit, newdata)` returns on the GLM-type
  routes (and what `fitted(fit)` returned before the default changed). The
  single `relmat(1 | g)`, `animal(1 | g)` and dense `phylo(1 | g)` Gaussian fits and the
  `phylo(1 | g)` Laplace fits of Poisson, Binomial, BetaBinomial, NB2, Gamma and Beta
  keep their modes, so `fitted(fit)` is conditional there. So do the zero-inflated
  Poisson, zero-inflated NB2 and zero-truncated NB2 fits with one `(1 | g)` and the
  `CumulativeLogit` fit with `(1 | g) + (0 + x | g)`: `fitted(fit)` is
  `(1 - zi) * mu`, `mu / (1 - P(0))` or the expected category score, with the modes
  in `mu` or in the linear predictor, as drmTMB's `fitted()`. A route that does not keep
  the modes needed for the conditional value (for example a `relmat` or `animal` count
  fit, `ZeroOneBeta` with a structured random effect, bivariate fits with random effects,
  a random effect on `sigma` when the mean depends on `sigma` (except the `LogNormal`
  `sigma ~ ... + (1 | g)` fit, which keeps its `sigma` modes), the crossed
  adaptive-quadrature opt-in routes) refuses the
  default with a message that names `re = :zero`; it never returns population
  values under the conditional default. `residuals(fit)` is `y - fitted(fit)`
  and takes the same `re` keyword. `residuals(fit; type = :quantile)` and
  `sigma(fit)` also return the used rows only, so `fitted`, `residuals` and
  `sigma` all have length `nobs(fit)` (`simulate(fit)` still returns one value
  per input row). drmTMB has a second missing-data policy that returns full-length
  values with `NA` for the unused rows; DRModels.jl matches the default (drop)
  policy only, and the second one is a planned follow-up.
  For LogNormal and zero-inflated, hurdle or
  truncated counts both packages return the unconditional response mean `E[Y]` (`exp(μ + σ²/2)`, `(1 - π) μ`,
  `μ / (1 - P(0))`, `(1 - π) μ / (1 - P(0))`), and `predict(fit, newdata; type =
  :mean)` is the same quantity at new rows. A bivariate `LogNormal` fit follows the
  same rule per response (`Dict(:mu1 => …, :mu2 => …)`; drmTMB returns an `n × 2`
  matrix), and its `predict(type = :response)` is the log-scale location.
  `fit.means[:mu]` keeps the component mean.
- Missing responses in a bivariate fit: both packages drop a row with either
  response missing by default (drmTMB `miss_control(response = "drop")`, silent;
  DRModels.jl warns with the count). `missing = miss_control(response = "include")`
  keeps partly observed rows and fits the marginal likelihood of the observed
  response in both. Before this default was aligned, DRModels.jl kept those rows
  unasked and its logLik differed from drmTMB's by 15 to 30 units on the same data.

drmTMB's exact R spelling for per-parameter prediction is not asserted here.
The response-scale `predict` calls name the intended comparable operation; parity
is established only for the specific outputs and fixtures listed in the
[parity evidence](drmtmb-parity.md), not for every family or prediction type.

## Deliberate difference — the log-scale numerical guard

Both twins guard the `log(sigma)` linear predictor inside the likelihood so a
runaway scale cannot blow the objective up, and **both report the guarded value**:
`sigma(fit)`, `simulate` and the quantile residuals describe the scale the
likelihood actually scored, not a raw `exp(X_sigma * beta_sigma)` the model never
evaluated (drmTMB `drm_clamped_sigma_eta()`; DRModels.jl `_reported_sigma`).

The *bands* differ on purpose. drmTMB soft-clamps every scale family at
`logsigma_clamp = c(-12, 12, 3)`. DRModels.jl keeps each family's own band — NB2
±20, Gamma ±15, Beta ±15 (margin 3) — because those families carry
`size`/`shape`/`precision` `= exp(-2 * eta_sigma)`, where drmTMB's ±12 would
distort legitimate fits (a near-Poisson NB2 wants `eta_sigma` ≈ -15); and the
**Gaussian fixed-effects core is not clamped at all**, since its objective is
well behaved without one. So on a row where `|eta_sigma| > 12` the two twins'
likelihoods, optima and reported scales can differ — a documented divergence,
not a bug. A fit that reaches any of these bands is at a dispersion boundary:
the objective is flat there, the Wald SEs are untrustworthy, and the reported
scale is honest about what was scored rather than a reliable estimate.

## Naming rules to remember

- **Scale is `sigma`, never `tau`.** `bf(y ~ x, tau ~ x)` is rejected — use
  `sigma`. (Group-level phylo/spatial/study variances are reported as named
  covariance summaries, not as a residual scale.)
- **`rho12` is the bivariate *residual* correlation** between the two responses
  (on the `atanh` scale). It is only valid in the keyword `bf(mu1 = …, mu2 = …,
  rho12 = …)` form — not as a univariate parameter.
- **ML is the default.** REML is an option (the likelihoods are not comparable
  across different fixed-effect structures, so ML is used for model selection).

For the optional R-to-Julia route, see [Coming from R](coming-from-r.md).
It is limited to the documented admitted models; ordinary `drmTMB` use
continues to use the default R engine.
