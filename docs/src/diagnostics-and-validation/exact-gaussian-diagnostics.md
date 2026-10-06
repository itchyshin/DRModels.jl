# Checking Gaussian mixed models

Use this page after fitting a Gaussian model with one or more random effects.
It shows the routine checks to make before interpreting coefficients, variance
components, or confidence intervals.

For Gaussian responses, DRModels.jl can integrate many mean-axis random effects
exactly. Here, **exact** describes the likelihood calculation: it does not by
itself guarantee convergence, good identification, or reliable inference for a
particular dataset.

## Fit a small model

The example below gives each group its own intercept while allowing the response
mean to change with `x`.

```@example gaussian_diagnostics
using DRModels, Random
Random.seed!(20260920)

ngroups = 12
n_per_group = 8
group = repeat(1:ngroups; inner = n_per_group)
x = randn(length(group))
group_intercept = 0.6 .* randn(ngroups)
y = 1.0 .+ 0.5 .* x .+ group_intercept[group] .+ 0.4 .* randn(length(group))
dat = (; y, x, group)

fit = drm(
    bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1)),
    Gaussian();
    data = dat,
)
```

The fixed-effect slope estimates how the average response changes with `x`.
The group standard deviation describes variation among group intercepts, and
`sigma(fit)` describes the remaining within-group spread.

```@example gaussian_diagnostics
(fixed_effects = coef(fit, :mu),
 group_sd = re_sd(fit),
 residual_sd = first(sigma(fit)))
```

## Run the fit check

[`check_drm`](@ref) gathers the main numerical checks in one report.

```@example gaussian_diagnostics
diagnostics = check_drm(fit)
(converged = diagnostics.converged,
 max_abs_grad = diagnostics.max_abs_grad,
 covariance_complete = diagnostics.vcov_complete,
 covariance_positive_definite = diagnostics.vcov_posdef,
 ok = diagnostics.ok)
```

Read the fields together:

- `converged` says whether the optimiser reported success.
- `max_abs_grad` measures how close the fit is to a stationary point. Values
  close to zero are reassuring; also inspect `grad_source` because some model
  types cannot provide this check.
- `vcov_complete` says whether a complete coefficient covariance matrix is
  available. Some sparse phylogenetic fits provide only the fixed-effect block.
- `vcov_posdef` says whether the available covariance matrix is
  positive-definite. A failure often means that one direction is weakly
  identified or lies on a variance boundary.
- `ok` summarises the checks that are available for this fit. It is a numerical
  summary, not a test of biological plausibility or model adequacy.

## If a check fails

A failed check is a reason to investigate, not automatically a reason to discard
the model.

- A large gradient with `converged = false` suggests that optimisation stopped
  too early. Standardise continuous predictors and refit before changing the
  scientific model.
- A non-positive-definite covariance matrix can occur when a random-effect
  variance is estimated near zero or when predictors are strongly confounded.
  Inspect the fitted variance components and the design matrix.
- A missing covariance block means that Wald intervals are not available for
  every parameter. Do not turn absent uncertainty into a precise claim.

The [convergence guide](../model-guides/convergence.md) gives a fuller decision
path. For phylogenetic models, also read the
[phylogenetic-effects tutorial](../tutorials/phylogenetic-models.md), whose
example shows the required tree input.

## What these checks establish

These checks assess the numerical result returned for this dataset. They do not
establish frequentist interval coverage, prove that the model is scientifically
appropriate, or extend Gaussian results to non-Gaussian models. Simulation or
bootstrap checks may still be needed for the quantity and sample size that
matter in your study.
