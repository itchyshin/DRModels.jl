# Improving convergence

!!! note "Status — Stable"
    Mirrors drmTMB's [Improving convergence](https://itchyshin.github.io/drmTMB/articles/convergence.html). A practical guide to checking and improving a fit in **DRModels.jl**, built around the [`check_drm`](@ref) diagnostic.

A distributional regression fits by maximum likelihood: `drm` runs an LBFGS
optimiser (with a robust mode-finder for the phylogenetic engine) and stores
whether it converged. Most models converge without intervention. This page is
for the ones that don't — and how to tell the difference between a real problem
and a model that is simply sitting at the edge of its parameter space.

## Step 1 — check the fit

Every fit carries a convergence flag and a covariance matrix for its
estimates (on most routes the inverse of the observed information).
[`check_drm`](@ref) bundles the useful diagnostics into one report:

```julia
using DRModels
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = dat)
check_drm(fit)
```

The report is a `NamedTuple`:

- `converged` — did the optimiser report success, and (for Gaussian fits) is the
  fit non-degenerate, as [`is_converged`](@ref) decides? This is still a
  stopping rule, not proof of a maximum (see below).
- `max_abs_grad` — `max|∇nll|` at the solution, the largest slope of the
  negative log-likelihood. Near a clean interior optimum this is ≈ 0; a large
  value means the optimiser stopped early.
- `grad_source` — how `max_abs_grad` was obtained. `:none` or `:unavailable`
  means no gradient could be computed, so `max_abs_grad` is `NaN` and the
  gradient was not checked at all.
- `vcov_posdef` — is the covariance positive-definite? When this is `false`,
  some parameters have no finite standard error (drmTMB's `sdreport` returns all
  `NaN` in exactly this situation).
- `min_eigval` / `cond` — the smallest eigenvalue and the condition number of the
  covariance. A near-zero `min_eigval` flags a direction the data barely
  constrain (a variance pinned near zero, two confounded coefficients).
- `ok` — `true` when converged, the gradient is small, and the covariance is PD.
  When `grad_source` is `:none` or `:unavailable`, `ok = true` means only
  "converged and PD": the gradient test was skipped.

!!! warning "What a `true` flag does not tell you"
    `converged = true` (and `is_converged(fit) = true`) means the optimiser
    stopped by its own rules. It does not prove the estimates are at a
    well-defined maximum. A fit can stop with a parameter at its limit and
    still report `true`: a variance estimated at zero, a Student-t `nu` that
    runs towards infinity (the data look Gaussian), or a `ZeroOneBeta`
    one-inflation part with no exact ones in the data.
    `is_converged` adds a check for a collapsed residual SD, but only for
    Gaussian fits. So read `max_abs_grad` and `grad_source` as well, and look
    at estimates that are very large on the link scale. Work to make the flag
    stricter is tracked in
    [DRModels.jl#969](https://github.com/itchyshin/DRModels.jl/issues/969) and
    [DRModels.jl#949](https://github.com/itchyshin/DRModels.jl/issues/949).

## Step 2 — read the failure mode

A non-`ok` report is information, not necessarily an error.

### A variance at the boundary

If a random-effect SD or a scale parameter is estimated near zero, the
likelihood flattens in that direction: `min_eigval` is tiny and the gradient may
not reach zero. This is often the *correct* MLE — the data carry little of that
variance component. The Wald standard errors of the other parameters can
usually still be used; the boundary direction itself has no meaningful Wald
standard error. Prefer [`confint`](@ref)`(fit; method = :profile)` there: the
profile interval does not rely on the quadratic Wald approximation, which
breaks down at a boundary. Its likelihood-ratio calibration is still
asymptotic, so it is not exact either.

This applies to variance components estimated at zero. A *shape* parameter at
its limit behaves differently: for Student-t `nu` running towards infinity,
the fit can report a standard error of 0 and a zero-width Wald interval for
`nu` instead of an undefined one
([DRModels.jl#969](https://github.com/itchyshin/DRModels.jl/issues/969)). Do
not use that interval; a large `nu` means the data look Gaussian, and a
profile or bootstrap interval, or a Gaussian refit, is the better check.

### Confounded predictors

A near-singular covariance with a large `cond` usually means two columns of a
design matrix are nearly collinear (e.g. an intercept and a centred covariate
that is almost constant). Re-centre or drop one; refit.

### Stopped early

If `converged` is `false` but `max_abs_grad` is still sizable, the optimiser
probably ran out of iterations rather than hitting a boundary. `drm()` does not
yet let you raise the iteration limit or supply starting values
([DRModels.jl#948](https://github.com/itchyshin/DRModels.jl/issues/948)), and
`g_tol` will not help: it is the gradient size at which the optimiser may
stop, so a smaller value makes stopping harder, not easier. Instead:

- centre and scale continuous covariates and refit (Step 3 below);
- simplify the model, for example drop a `sigma` covariate or a random slope,
  and add terms back one at a time;
- compare with drmTMB, whose `drm_control()` does accept iteration limits and
  starting values.

## Step 3 — scale and standardise

Most convergence trouble is conditioning. Two cheap fixes:

- **Standardise covariates.** Predictors on wildly different scales make the
  Hessian ill-conditioned. Centre and scale continuous covariates before fitting.
- **Use ML for model comparison.** ML is the default and is comparable across
  fixed-effect structures; REML likelihoods are not. Keep ML when selecting
  models with [`aic`](@ref) / [`bic`](@ref).

## The phylogenetic engine

The verified q=4 phylogenetic location–scale engine has its own mode-finder
(sparse augmented-state Laplace with an exact O(p) gradient). Two notes specific
to it:

- **Off-diagonal initialisation.** Starting the cross-covariance Cholesky away
  from zero (the `lc3`/`lc7` entries) avoids a saddle near the diagonal start.
- **The Watanabe-singular boundary.** When a per-axis phylogenetic variance is
  near zero the model sits at a singular point; the optimiser flags the plateau
  rather than chasing a flat ridge. This matches drmTMB's behaviour (it reports
  "false convergence") — both engines are correct; the geometry, not the
  optimiser, is the limiting factor.
