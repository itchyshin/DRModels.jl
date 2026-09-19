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

[`predict`](@ref) evaluates the fitted **mean** at a `newdata` column table. By
default it returns the response scale; pass `type = :link` for the linear
predictor.

```@example postfit
nd = (; x = [-1.0, 0.0, 1.0])
predict(fit, nd)                 # μ̂ at x = -1, 0, 1  (response scale)
```

### Standard errors (the delta method)

Add `se = true` for delta-method standard errors — the glmmTMB/drmTMB `se.fit`.
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
`predict_parameters(fit, data)`:

```@example postfit
mp = marginal_parameters(fit)
first(mp[:mu], 3), first(mp[:sigma], 3)
```

## Residuals: response and quantile

[`residuals`](@ref) defaults to response residuals (`y − μ̂`). For a *distribution-aware*
diagnostic — the DHARMa/glmmTMB **randomized quantile residual** — pass
`type = :quantile`. Under a correctly specified model these are standard normal,
which makes a QQ-plot or a simple moment check interpretable even for
non-Gaussian families:

```@example postfit
rq = residuals(fit; type = :quantile)
(mean = mean(rq), sd = std(rq))         # ≈ (0, 1) when the model fits
```

!!! tip "Why quantile residuals"
    Raw `y − μ̂` residuals are misleading when the variance changes with the
    mean (exactly the location–scale case here) or for discrete responses.
    Quantile residuals fold each observation through its own fitted CDF, so a
    well-specified model always yields ≈ N(0, 1) — the scale is the same across
    families. It is implemented for Gaussian and Poisson today; support for
    other families is planned.

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
