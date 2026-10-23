# Checking and using fitted models

!!! note "Status — Stable (Gaussian post-fit + inference)"
    Mirrors drmTMB's [Checking and using fitted models](https://itchyshin.github.io/drmTMB/articles/model-workflow.html).
    **In DRModels.jl today:** coefficient extraction, Wald standard errors, Wald
    **and profile-likelihood** confidence intervals, fitted values, residuals,
    `predict` (new data), `simulate`, and **parametric bootstrap** intervals
    (`bootstrap_ci`).

Once you have a fit, pull coefficients and quantify their uncertainty.

```@example wf
using DRModels, Random
Random.seed!(5)
n = 1500
x = randn(n)
y = 0.4 .+ 0.7 .* x .+ exp.(-0.2 .+ 0.3 .* x) .* randn(n)
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))

coef(fit, :mu)        # mean coefficients
stderror(fit)         # Wald standard errors (all coefficients)
```

Wald confidence intervals for every coefficient (drmTMB's default interval
method), one row per coefficient:

```@example wf
confint(fit; level = 0.95)
```

Each row carries its parameter (`:mu` / `:sigma`), coefficient name, point
estimate, and interval bounds. Intervals are on each parameter's working scale —
`μ` on the response scale, `σ` on `log σ` — so for a residual-SD ratio you
exponentiate the `σ` bounds.

For a **profile-likelihood** interval (drmTMB's `method = "profile"`), pass
`method = :profile`. It inverts the likelihood-ratio statistic — re-optimising
the nuisance parameters at each fixed value — so it can reflect an asymmetric
likelihood shape where Wald uses a quadratic approximation. Its likelihood-ratio
calibration is still asymptotic. Use Wald for speed; consider a profile when a
parameter's likelihood is skewed (for example, a scale or variance term):

```@example wf
confint(fit; method = :profile)
```

!!! tip "What scale is the interval on?"
    A `σ` row gives an interval for a `log σ` coefficient. `exp(lower)` and
    `exp(upper)` give the interval for the multiplicative effect on the residual
    SD. See [Which scale are you modelling?](which-scale.md).

## Fitted values and residuals

```@example wf
ŷ = fitted(fit)            # fitted means (Xβ̂)
res = residuals(fit)       # observed − fitted
(n = length(res), sum_resid = round(sum(res), digits = 6))
```

For a bivariate model, `fitted` / `residuals` return one vector per response,
keyed `Dict(:mu1 => …, :mu2 => …)`.

Predict the mean on **new** data (population level — random/structured effects
integrated out):

```@example wf
predict(fit, (; x = [-1.0, 0.0, 1.0]))    # μ̂ at three new x values
```

## Simulating from a fitted model

`simulate` draws a parametric replicate — the building block of a parametric
bootstrap:

```@example wf
y_rep = simulate(fit; rng = MersenneTwister(1))   # one replicate response vector
length(y_rep)
```

## Bootstrap confidence intervals

`bootstrap_ci` automates the parametric bootstrap — simulate `B` replicates,
refit each, and take percentile intervals. It takes the same arguments as `drm`
(plus `B`), since it refits internally:

```julia
bootstrap_ci(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian();
             data = dat, B = 500, level = 0.95)
```

Returns the same `(param, coef, estimate, lower, upper)` rows as `confint`. Use
Wald (`confint`) for speed; use a bootstrap to see how interval estimates vary
when data are repeatedly simulated and refitted under the fitted model. A
parametric bootstrap therefore still depends on that model's assumptions.

For longer jobs, use `bootstrap_result` to keep the audit trail:

```julia
bres = bootstrap_result(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian();
                        data = dat, B = 500, threads = true,
                        failures = :skip)
(bres.attempted, bres.used, bres.failed)
```

## See also

- [Get started](../getting-started.md) · [When variance carries signal](../tutorials/location-scale.md)
- Profile and bootstrap intervals, `predict`, and `simulate` are all shown
  above; for their evidence status see [Evidence & limits](../capabilities.md).
