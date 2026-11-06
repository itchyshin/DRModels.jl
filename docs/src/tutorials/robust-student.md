# Robust continuous responses

!!! note "Status — Stable"
    Mirrors drmTMB's [Robust continuous responses](https://itchyshin.github.io/drmTMB/articles/robust-student.html).
    **In DRModels.jl today:** the Student-t family `Student()` — a formula per
    parameter for the location `μ`, scale `σ`, and degrees of freedom `ν`. Fixed
    effects, maximum likelihood.

When a few observations sit far from the trend, a Gaussian fit chases them — the
mean tilts and `σ` inflates. The **Student-t** family gives the residuals heavier
tails, so outliers are downweighted instead of dominating. The shape parameter
`ν` (degrees of freedom) controls how heavy: small `ν` is very robust, and
`ν → ∞` returns to Gaussian.

## Fitting a Student-t model

The code below also loads the `Distributions` package to simulate data. If you have not added it to your Julia environment, run `using Pkg; Pkg.add("Distributions")` once first.

Give each parameter its own formula, exactly like the Gaussian model — `nu ~ 1`
estimates a single degrees-of-freedom value:

```@example student
using DRModels, Random
using Distributions: TDist
Random.seed!(20260614)

n = 2000
x = randn(n)
# location-scale t errors (heavy-tailed): true ν = 5
y = 0.5 .+ 0.7 .* x .+ 0.8 .* rand(TDist(5.0), n)
dat = (; y, x)

fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = dat)
coef(fit, :mu)              # location coefficients
```

`σ` is on the log scale, so `exp` returns it. `ν` uses a **logm2** link,
`ν = 2 + exp(η)`, which keeps `ν > 2` so the variance stays finite — read it back
with `2 + exp(...)`, not `exp(...)`:

```@example student
exp(coef(fit, :sigma)[1])       # residual scale
2 + exp(coef(fit, :nu)[1])      # degrees of freedom, ν = 2 + exp(η)  (≈ 5 ⇒ heavy tails)
```

As `ν` grows, the Student-t likelihood approaches its Gaussian limit, so very
large values can be weakly identified. If the reported standard error for `nu`
is zero or non-finite, treat its Wald interval as unreliable even if the
optimizer reports convergence. Compare with a Gaussian fit when that simpler
model answers the same question. A profile interval is useful only when the
target is available and its evidence supports the intended interpretation.

!!! tip "Scale, not variance"
    For the t the residual *variance* is `σ² · ν/(ν−2)` (and is undefined for
    `ν ≤ 2`). `σ` here is the scale parameter, not the SD — quote it as such when
    comparing to a Gaussian fit.

## See also

- [When variance carries signal](location-scale.md) — the Gaussian location–scale model.
- [What can I fit today?](../model-guides/model-map.md) — the family/feature matrix.
