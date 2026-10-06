# Improving convergence

!!! note "Status — Stable"
    Mirrors drmTMB's [Improving convergence](https://itchyshin.github.io/drmTMB/articles/convergence.html). A practical guide to checking and improving a fit in **DRModels.jl**, built around the [`check_drm`](@ref) diagnostic.

A distributional regression fits by maximum likelihood: `drm` runs an LBFGS
optimiser (with a robust mode-finder for the phylogenetic engine) and stores
whether it converged. Most models converge without intervention. This page is
for the ones that don't — and how to tell the difference between a real problem
and a model that is simply sitting at the edge of its parameter space.

## Step 1 — check the fit

Every fit carries a convergence flag and an observed-information covariance.
[`check_drm`](@ref) bundles the useful diagnostics into one report:

```julia
using DRModels
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = dat)
check_drm(fit)
```

The report is a `NamedTuple`:

- `converged` — did the optimiser report success?
- `max_abs_grad` — `max|∇nll|` at the solution. Near a clean interior optimum
  this is ≈ 0; a large value means the optimiser stopped early.
- `vcov_posdef` — is the covariance positive-definite? When this is `false`,
  some parameters have no finite standard error (drmTMB's `sdreport` returns all
  `NaN` in exactly this situation).
- `min_eigval` / `cond` — the smallest eigenvalue and the condition number of the
  covariance. A near-zero `min_eigval` flags a direction the data barely
  constrain (a variance pinned near zero, two confounded coefficients).
- `ok` — `true` when converged, the gradient is small, and the covariance is PD.

## Step 2 — read the failure mode

A non-`ok` report is information, not necessarily an error.

### A variance at the boundary

If a random-effect SD or a scale parameter is estimated near zero, the
likelihood flattens in that direction: `min_eigval` is tiny and the gradient may
not reach zero. This is often the *correct* MLE — the data carry little of that
variance component. The other parameters still have valid Wald standard errors;
only the boundary direction is undefined. Prefer
[`confint`](@ref)`(fit; method = :profile)` there: the profile interval is exact
under the likelihood-ratio statistic where the quadratic Wald approximation
breaks down at a boundary.

### Confounded predictors

A near-singular covariance with a large `cond` usually means two columns of a
design matrix are nearly collinear (e.g. an intercept and a centred covariate
that is almost constant). Re-centre or drop one; refit.

### Stopped early

If `converged` is `false` but `max_abs_grad` is still sizable, the optimiser ran
out of iterations rather than hitting a boundary. Tighten the gradient tolerance
and refit:

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian();
          data = dat, g_tol = 1e-10)
```

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
