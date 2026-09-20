# When variance carries signal, Part 1: location–scale

!!! note "Status — Stable"
    Mirrors drmTMB's [When variance carries signal, Part 1](https://itchyshin.github.io/drmTMB/articles/location-scale.html).
    **In DRModels.jl today:** the univariate Gaussian location–scale model fits by ML.
    Continue with [Part 2: location–scale–scale](location-scale-scale.md) when the
    question moves from the residual SD to the SD of a random effect.

Sometimes the science is in the *spread*, not the mean. A classic case: two
conditions with the **same** average response but different variability. A
mean-only model is blind to that; a location–scale model sees it.

## A variance that moves while the mean stays put

```@example ls
using DRModels, Random
Random.seed!(7)

n = 400
group = rand(0:1, n)
# identical mean (0), but the residual SD is larger in group 1:
y = 0.0 .+ exp.(-0.2 .+ 0.8 .* group) .* randn(n)
dat = (; y, group = Float64.(group))

fit = drm(bf(@formula(y ~ 1), @formula(sigma ~ group)), Gaussian(); data = dat)
coef(fit, :sigma)        # (Intercept), group — on log σ
```

The `group` coefficient on `log σ` is ≈ `0.8`. On the response scale that is a
residual-SD ratio:

```@example ls
exp(coef(fit, :sigma)[2])     # ≈ how many times larger the SD is in group 1
exp(2 * coef(fit, :sigma)[2]) # the residual-VARIANCE ratio
```

!!! tip "Name the scale"
    σ coefficients are on `log σ`. Effects on `log σ²` are **2×** the effects on
    `log σ` — quote the scale explicitly when comparing to DHGLM / O'Dea-style
    variance models.

Working in R? The [drmTMB companion tutorial](https://itchyshin.github.io/drmTMB/articles/location-scale.html)
reports the same explicit SD and variance ratios with R syntax. The two packages
have distinct APIs, fitting implementations, and route-specific evidence, so
follow the diagnostics and inference guidance for the package you fit with.

## Does the scale model earn its keep?

Compare the location–scale fit to a constant-σ fit by log-likelihood:

```@example ls
fit0 = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data = dat)
loglik(fit) - loglik(fit0)    # gain from letting σ depend on group
```

A large positive gain says the variance structure is real signal. Quantify the
σ effects with [`confint`](../model-guides/model-workflow.md) (Wald or profile),
or turn the comparison into a formal test with `lrtest` / `aicc` — see
[Prediction, residuals & model comparison](../diagnostics-and-validation/prediction-and-postfit.md).

## Random dispersion: a scale that varies by group

When the *spread* itself varies across many groups (sites, individuals,
studies), put a **random intercept on `σ`** rather than a fixed level per group.
Because the random effect enters σ nonlinearly there is no closed-form marginal,
so DRModels.jl integrates each group's effect out with per-group **Gauss–Hermite
quadrature**. drmTMB uses Laplace: both target the same marginal model, but
their numerical approximations need not be identical.

```@example ls
Random.seed!(13)
G = 40; m = 25; ng = G * m
grp = repeat(1:G, inner = m)
bg = 0.5 .* randn(G)                    # group-level log-σ deviations, SD 0.5
yr = exp.(log(0.5) .+ bg[grp]) .* randn(ng)
datre = (; y = yr, grp)

fitre = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1 + (1 | grp))), Gaussian(); data = datre)
re_sd(fitre)[:grp_logsigma]  # recovered log-σ group-effect SD (≈ 0.5)
```

`re_sd` returns the scale-RE SD; `coef(fitre, :sigma)` is the population
(group-average) `log σ`.

## See also

- [Part 2: location–scale–scale](location-scale-scale.md) — predictors on the
  SD of a random effect (iid and phylogenetic).
- [Get started](../getting-started.md) — your first fit.
- [Changing residual coupling with rho12](bivariate-coscale.md) — two responses.
