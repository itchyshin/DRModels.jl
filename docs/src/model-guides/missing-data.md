# Handling missing data

!!! note "Status — Partial"
    Related drmTMB article: [Handling missing data](https://itchyshin.github.io/drmTMB/articles/missing-data.html).
    drmTMB's missing-data routes are much broader than DRModels.jl's today.
    **In DRModels.jl today:** rows with a missing response are dropped with a
    warning; `drm_listwise` drops incomplete rows explicitly; and an
    experimental `mi()` route models one or two missing predictors for a
    Gaussian response. There is no general multiple imputation and no
    full-information route for missing responses in bivariate models.

Real data sets have gaps. What DRModels.jl does with them depends on whether
the gap is in the **response** (the variable on the left of `~`) or in a
**predictor** (a variable on the right).

## Missing responses

If some response values are `missing` (or `NaN`), `drm` fits the rows whose
response is observed and warns how many rows it dropped. This is the same
default as glmmTMB's `na.action`. Dropping these rows gives unbiased estimates
when missingness depends only on the predictors in the model, not on the
unobserved response value itself.

```@example missing
using DRModels, Random
Random.seed!(3)

n = 60
x = randn(n)
y = 1 .+ 0.5 .* x .+ 0.4 .* randn(n)
y_obs = Union{Missing,Float64}[i % 10 == 0 ? missing : y[i] for i in 1:n]   # 6 gaps

fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
          data = (; y = y_obs, x))
coef(fit, :mu)
```

For a bivariate model, a row is dropped when either response is missing. That
discards the observed half of each incomplete pair.

## Missing predictors

A `missing` predictor value makes `drm` stop with an error, so nothing is
dropped without your knowledge. You then have two choices.

**Complete-case analysis.** [`drm_listwise`](@ref) removes every row with a
missing or non-finite value in any column the model uses, warns how many rows
it dropped and which columns caused it, and returns data ready for `drm`. The
example turns the warning off and reports the row counts instead:

```@example missing
z = randn(n)
z_obs = Union{Missing,Float64}[i % 7 == 0 ? missing : z[i] for i in 1:n]
dat = (; y, x, z = z_obs)

form  = bf(@formula(y ~ x + z), @formula(sigma ~ 1))
clean = drm_listwise(form, dat; verbose = false)   # verbose = true (default) warns
(rows_before = n, rows_after = length(clean.y))
```

```@example missing
fit_cc = drm(form, Gaussian(); data = clean)
coef(fit_cc, :mu)
```

Complete-case analysis is unbiased when the rows are missing completely at
random, and it wastes the observed values in the dropped rows.

**Model the missing predictor (experimental).** For a Gaussian response,
writing `mi(x)` in the mean formula and giving a model for `x` integrates the
missing `x` values out of the likelihood instead of dropping their rows. This
supports one Gaussian or binary predictor, or two independent Gaussian
predictors, with fixed effects only. See
[Modelled missing predictors](@ref joint-predictor-formula) for the syntax,
a worked example, and the full list of limits.

## What is not available yet

- multiple imputation, or an `na.action`-style option to choose the
  behaviour;
- full-information likelihood that keeps the observed half of an incomplete
  bivariate pair;
- missing-predictor models for non-Gaussian responses, for count, ordinal or
  proportion predictors, or together with random or structured effects (drmTMB
  supports several of these);
- missing-predictor models through the R bridge beyond the narrow route above.

If you need one of these, fit the model in drmTMB, or impute outside the
package and analyse each imputed data set separately.

## See also

- [What is tested](../capabilities.md), section "Absent / out-of-scope"
- [Checking and using fitted models](model-workflow.md)
