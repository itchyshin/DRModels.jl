# Prediction, residuals & model comparison

!!! note "Status — Stable"
    The post-fit surface a fitted `drm` model exposes: prediction (point +
    standard error, per parameter, on a grid), quantile residuals, and
    model-comparison helpers. Mirrors the `predict` / `residuals` / `anova`
    verbs you reach for after a drmTMB fit.

A fit is the beginning, not the end. Once `drm` returns, you usually want to
**predict** at new covariate values, **check** the model with residuals, and
**compare** it against a simpler one. This page walks the whole surface on one
small location–scale model.

```@example postfit
using DRModels, Random, Statistics
Random.seed!(11)

n = 500
x = randn(n)
# mean μ = 1 + 0.5x ; log-scale logσ = -0.2 + 0.6x  (the spread grows with x)
y = 1.0 .+ 0.5 .* x .+ exp.(-0.2 .+ 0.6 .* x) .* randn(n)
dat = (; y, x)

fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = dat)
coef(fit, :mu), coef(fit, :sigma)
```

## Predicting the mean at new data

[`predict`](@ref) evaluates the fitted **mean** at a `newdata` column table.
`type = :response` (the default) returns `invlink(Xβ̂)` with every random
effect set to 0; pass `type = :link` for the linear predictor `Xβ̂`. Those
random or structured effects are not integrated out, so on a random-effect
fit this is the prediction for an average group, not the mean averaged over
groups.

```@example postfit
nd = (; x = [-1.0, 0.0, 1.0])
predict(fit, nd)                 # μ̂ at x = -1, 0, 1  (response scale)
```

### Standard errors (the delta method)

Add `se = true` for delta-method standard errors — the glmmTMB `se.fit`
convention (drmTMB's `predict()` has no standard-error option). They carry the
uncertainty in `β̂` only.
You get back a `NamedTuple` `(; prediction, se)`:

```@example postfit
ps = predict(fit, nd; type = :link, se = true)
ps.se                            # SE of the linear predictor at each new row
```

On the response scale the SE is chained through the inverse-link derivative
automatically (for the Gaussian identity link the two coincide):

```@example postfit
predict(fit, nd; type = :response, se = true).se
```

## Predicting every distributional parameter

A distributional model has more than a mean. [`predict_parameters`](@ref)
returns **each** parameter at the new data — here `μ` *and* `σ`:

```@example postfit
predict_parameters(fit, nd)      # Dict(:mu => …, :sigma => …), response scale
```

With `se = true` each entry becomes a `(; value, se)` pair:

```@example postfit
pp = predict_parameters(fit, nd; se = true)
pp[:sigma].value, pp[:sigma].se  # fitted σ and its SE at x = -1, 0, 1
```

## Sweeping a predictor: `prediction_grid`

To trace a parameter across a covariate, build a grid with
[`prediction_grid`](@ref) — it sweeps the named predictor(s) over a range and
holds everything else at a reference value, then composes straight into
`predict_parameters`:

```@example postfit
grid = prediction_grid((; x = dat.x), x = range(-2, 2; length = 5))
predict_parameters(fit, grid)[:sigma]   # σ̂ rising across the x sweep
```

## In-sample fitted parameters

For the per-observation fitted parameters at the **training** data there is a
cheap accessor that reads straight from the fit (no recomputation) —
[`marginal_parameters`](@ref). In-sample it equals
`predict_parameters(fit, data)` for the mean and `sigma` of the GLM-type families
(Gaussian, Poisson, NegBinomial2, Gamma, Beta, Binomial), as here (the
[`predict_parameters`](@ref) docstring lists the exceptions):

```@example postfit
mp = marginal_parameters(fit)
first(mp[:mu], 3), first(mp[:sigma], 3)
```

## Residuals: response and quantile

[`residuals`](@ref) defaults to response residuals (`y − μ̂`). For a *distribution-aware*
diagnostic — the DHARMa/glmmTMB **randomized quantile residual** — pass
`type = :quantile`. For a correctly specified model without random effects,
like this one, they are approximately standard normal, which makes a QQ-plot or
a simple moment check interpretable even for non-Gaussian families:

```@example postfit
rq = residuals(fit; type = :quantile)
(mean = mean(rq), sd = std(rq))         # ≈ (0, 1) when the model fits
```

!!! tip "Why quantile residuals"
    Raw `y − μ̂` residuals are misleading when the variance changes with the
    mean (exactly the location–scale case here) or for discrete responses.
    Quantile residuals fold each observation through its own fitted CDF, so a
    well-specified model without random effects yields ≈ N(0, 1), on the same
    scale for every family. They are implemented for every family except
    Tweedie and SkewNormal; for zero-inflated and hurdle count fits the CDF is
    that of the whole mixture, zero part included. Random-effect and `temporal()` fits judge each
    observation against a different reference (a random intercept integrated
    out, or the fitted temporal path), so what "≈ N(0, 1)" means changes;
    [`residuals`](@ref) lists what each kind of fit returns.

!!! warning "Accuracy limit when the random-effect variance is large (#1103)"
    Where a random intercept is integrated out by the 32-node Gauss–Hermite
    mixture (non-Gaussian `(1 | g)` on the mean, a Gaussian `sigma ~ 1 + (1 | g)`
    intercept, and structured effects that keep the mixture), the conditional CDF
    becomes a near-step in the node variable once σ̂_b ≫ σ̂, and 32 nodes stop
    resolving it. Measured for a Gaussian mean intercept (max |Φ⁻¹(mixture) −
    exact z| over |z| ≤ 4): about 1e-6 at σ_b/σ = 2, 6e-3 at 4, 0.02 at 5,
    0.12 at 10 and 0.21 at 20. Read the residuals from those routes as
    approximate when σ̂_b/σ̂ is above about 3. Gaussian `(1 | g)` on the mean and
    Gaussian / LogNormal `phylo(1 | g)` already use the closed-form marginal and
    are not affected. Separately, every continuous family computes
    `Φ⁻¹(clamp(Φ(z), eps, 1 − eps))`, which loses upper-tail precision (error
    about 6e-6 at z = 7 and 8e-3 at z = 8) and caps |r| at about 8.13.

!!! note "Quantile residuals with a structured mean: DRModels.jl vs drmTMB"
    For a fit whose mean has a `phylo()`, `relmat()` or `animal()` random
    intercept, the two packages do not return the same numbers, although the
    fits agree (same log-likelihood to about 6e-8 on the checked Gaussian
    `phylo(1 | species)` model).

    | Convention | Residual | Who returns it |
    |---|---|---|
    | marginal | `(y − Xβ̂) / √(σ̂² + σ̂_b² C_ii)` | DRModels.jl (default Gaussian `phylo(1 \| g)` route) |
    | conditional | `(y − Xβ̂ − û) / σ̂` | drmTMB's `residuals(type = "quantile")` code |
    | fixed-effect only | `(y − Xβ̂) / σ̂` | what drmTMB's residual help text describes |

    On that model the two engines agree to within 3.5e-9 under either
    convention, so the gap is the convention, not the fit. Marginal residuals
    stay near N(0, 1) for each row, but rows that share ancestry are
    correlated, so read the whole set (a QQ plot, the SD) and do not hold each
    row to N(0, 1). Do not compare the two packages' quantile residuals number by number for
    these fits. [`residuals`](@ref) lists the convention for every other kind
    of fit. Which convention drmTMB should use is an open drmTMB-side decision
    ([#1101](https://github.com/itchyshin/DRModels.jl/issues/1101)).

## Does the extra structure earn its keep?

Fit a simpler model with a **constant** scale and compare. [`lrtest`](@ref)
(alias [`anova`](@ref)) runs the nested likelihood-ratio test; it returns a
`NamedTuple` `(; statistic, dof, pvalue)`:

```@example postfit
reduced = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = dat)
lrtest(reduced, fit)             # is σ ~ x worth the one extra parameter?
```

A small p-value supports the moving-scale term within these nested, checked
candidate models. Also inspect the estimated scale change, residual diagnostics,
and whether that effect answers the biological question. The information
criteria agree — lower is better, and [`aicc`](@ref) is the small-sample-corrected AIC:

```@example postfit
(aic = aic(fit),   bic = bic(fit),   aicc = aicc(fit)),
(aic = aic(reduced), bic = bic(reduced), aicc = aicc(reduced))
```

## Refitting: `update`

[`update`](@ref) re-fits with a new formula, reusing the original family — handy
for building a comparison ladder. (A `DrmFit` does not retain its data, so pass
`data` again.)

```@example postfit
mean_only = update(fit, bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)); data = dat)
length(coef(mean_only)) == length(coef(reduced))
```

## See also

- [When variance carries signal](../tutorials/location-scale.md) — the model fit here, in depth.
- [Rosetta (R ↔ Julia)](../rosetta.md) — the verb-by-verb drmTMB mapping.
- [Model fitting & post-fit reference](../reference/model-fitting-and-postfit.md) — every accessor's docstring.
