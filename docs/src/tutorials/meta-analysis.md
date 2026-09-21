# Mean effects and residual heterogeneity

!!! note "Status — Stable (univariate and row-paired bivariate)"
    Mirrors drmTMB's [Mean effects and residual heterogeneity](https://itchyshin.github.io/drmTMB/articles/meta-analysis.html).
    **This tutorial uses the univariate route:** Gaussian meta-analysis with
    **known** per-study sampling variances via `meta_V(v)`, plus estimated
    between-study heterogeneity τ (the `σ` parameter). For two outcomes,
    DRModels.jl also accepts one known 2×2 sampling-covariance block per study
    through `meta_vcov_bivariate(...)` and the `V =` fit keyword. Arbitrary
    cross-study covariance, REML with `V`, and `V` combined with structured
    effects are not supported; see the [bivariate guide](../model-guides/meta-analysis.md#Two-responses-and-now-two-correlations).

In meta-analysis each study reports an effect `y_i` with a **known** sampling
variance `v_i`. The model separates that known measurement uncertainty from the
**between-study heterogeneity** τ you want to estimate:

```math
y_i \sim \mathcal{N}(x_i^\top\beta,\; v_i + \tau^2).
```

Flag the known-variance column with `meta_V(v)` in the mean formula; the residual
heterogeneity τ is the `σ` parameter (`sigma ~ 1` for a single heterogeneity).
`meta_V(v)` is the current marker — the older `meta_known_V` is **deprecated** and
kept only as a parity stub.

## Random-effects meta-analysis (intercept only)

With no moderators the mean formula is `y ~ 1 + meta_V(v)`: a single pooled effect
on top of the known per-study variances plus estimated heterogeneity. This is the
classic random-effects meta-analysis.

```@example meta
using DRModels, Random
Random.seed!(20260603)

k = 300
v = (0.2 .+ 0.6 .* rand(k)) .^ 2     # KNOWN per-study sampling variances
μ = 0.4                              # true pooled effect (to recover)
τ = 0.4                             # between-study heterogeneity (to recover)
y = μ .+ τ .* randn(k) .+ sqrt.(v) .* randn(k)
dat = (; y, v)

fit0 = drm(bf(@formula(y ~ 1 + meta_V(v)), @formula(sigma ~ 1)), Gaussian(); data = dat)
coef(fit0, :mu)[1]                  # pooled estimate ≈ 0.4
```

The between-study heterogeneity τ is the `σ` intercept (on the log scale), so
exponentiate to read it back on the SD scale:

```@example meta
exp(coef(fit0, :sigma)[1])          # τ ≈ 0.4
```

## Meta-regression (adding a moderator)

Add study-level covariates to the mean formula to explain part of the
heterogeneity — `y ~ x + meta_V(v)` is meta-regression:

```@example meta
x = randn(k)                         # a study-level moderator
β = [0.2, 0.5]                       # intercept + slope (to recover)
yr = β[1] .+ β[2] .* x .+ τ .* randn(k) .+ sqrt.(v) .* randn(k)
datr = (; y = yr, x, v)

fit = drm(bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1)), Gaussian(); data = datr)
coef(fit, :mu)                       # overall intercept + moderator slope
```

The residual (unexplained) heterogeneity, after conditioning on the moderator, is
again the `σ` intercept:

```@example meta
exp(coef(fit, :sigma)[1])            # residual τ
```

Wald intervals via [`confint`](../model-guides/model-workflow.md) apply as usual to
both the mean (β) and σ (log τ) coefficients.

!!! note "Known V is not residual σ"
    `meta_V(v)` is *supplied* measurement uncertainty; τ (the `σ` parameter) is
    *estimated* heterogeneity. See [Which scale are you modelling?](../model-guides/which-scale.md).
