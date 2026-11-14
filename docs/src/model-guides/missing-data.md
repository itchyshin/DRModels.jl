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

`missing = miss_control(response = "include")` keeps the partly observed rows of a bivariate
Gaussian fit and uses the likelihood of the response that was observed in each of them.

**Missing at random.** This is valid when the values are missing at random. In plain
words: once the observed responses and the predictors in the model are taken into account,
whether a value is missing tells you nothing more about what that value would have been. It
fails when the missingness depends on the missing value itself (for example, the largest
`y1` values are the ones that were not recorded), or on something related to the response
that is not in the model. Neither the default (dropping the rows) nor `"include"` protects
against that. The difference is what the check involves: dropping the rows is valid only if missingness is also unrelated
to the other response of the same row, while `"include"` does not need that, because it uses the observed response of
those rows. Both still rely on the missing-at-random assumption.

**REML with kept rows.** On the bivariate `(1 | p | g)` route, `method = :REML` together with
`response = "include"` fits the restricted likelihood of the observed entries. drmTMB 0.8.1
refuses that combination ("`REML` is not implemented with explicit missing-data engines yet"),
so this goes beyond drmTMB and there is no R reference to compare with. It was checked
against a dense computation of the restricted likelihood of the observed entries
(`test/test_biv_re_include.jl`, agreement 1e-8) and against `nlme::lme` REML fits of the same model
written in long format, one row per observed entry (`test/test_biv_re_include_nlme.jl`).
Three safeguards apply:

- The fixed-effect design of the observed entries of each response must have full column rank.
  A column that is constant, or equal to another column, on the rows where that response is
  observed (a factor level seen only in rows with a missing response, a covariate that is constant there,
  a level whose rows all have weight 0) is refused by name. The message names `REML`, the
  response, the aliased columns and the options, which depend on the cause. Remove or recode the term,
  or use `method = :ML`, always works. Taking the rows with a missing response out of `data` (with a freshly
  built formula) helps only for a factor level seen in those rows alone; a covariate that is constant on the
  observed rows stays constant after the rows are removed. A formula object reused after a fit on other
  data keeps that data's factor levels, so build it afresh. A response with fewer observed
  entries than columns is refused with its own message, also under ML.
- A REML log-likelihood is comparable only between fits with the same fixed-effect design and the
  same used entries. A fit that keeps partly observed rows is not comparable with a complete-case fit
  of the same data (they use different entries), and REML values for different fixed-effect formulas
  are not comparable either. Do not use REML log-likelihoods, AIC, or likelihood-ratio tests to compare
  models with different fixed effects; refit with `method = :ML` for that.
- Rescaling a predictor changes the REML log-likelihood. Multiplying a fixed-effect column by `s` leaves the
  fit unchanged (the coefficient is divided by `s`) but shifts the REML log-likelihood by `-log(s)`, or by
  minus the sum of `log(s)` over all rescaled columns. A fit with a standardised predictor and a fit with the
  raw predictor therefore have different REML log-likelihoods; compare them only on the same scale.
  `method = :ML` values do not shift.
- The kept rows rely on the same missing-at-random assumption as above; nothing extra is checked for them.

## Missing predictors

A `missing` predictor value makes `drm` drop that row and warn how many rows it
dropped and which columns caused it (R's `na.omit`, as drmTMB does; #1089).
`nobs(fit)` counts the rows used, and prior `weights` are dropped with their
rows. A `NaN` predictor is not dropped: it is an input error. A `relmat`,
`animal` or `spatial` model whose levels the drop would reorder is refused by
name, because its matrix is matched to the levels by position. You have two
choices.

**Complete-case analysis.** Either let `drm` drop the rows, as above, or do it
yourself first. [`drm_listwise`](@ref) removes every row with a missing or
non-finite value in any column the model uses, warns how many rows it dropped
and which columns caused it, and returns data ready for `drm`. Use it when you
want the drop to be explicit, when predictors contain `NaN` or `Inf`, or when
you need the complete-case data for `bootstrap_ci`. The example turns the
warning off and reports the row counts instead:

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
