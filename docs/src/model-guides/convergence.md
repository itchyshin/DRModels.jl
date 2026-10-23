# Improving convergence

!!! note "Status — Stable"
    Mirrors drmTMB's [Improving convergence](https://itchyshin.github.io/drmTMB/articles/convergence.html). A practical guide to checking and improving a fit in **DRModels.jl**, built around the [`check_drm`](@ref) diagnostic.

A distributional regression fits by maximum likelihood: most `drm` routes run
an LBFGS optimiser (with a robust mode-finder for the phylogenetic engine) and store
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
and refit. To raise the iteration cap or supply starting values, pass one
`DRMControl` (the names follow drmTMB's `drm_control()`):

```julia
f = bf(@formula(y ~ x), @formula(sigma ~ x))
fit = drm(f, Gaussian(); data = dat, g_tol = 1e-10,
          control = DRMControl(maxit = 2000))

# Start from another fit's estimates, or try five starts and keep the best:
fit2 = drm(f, Gaussian(); data = dat, control = DRMControl(start = coef(fit)))
fit5 = drm(f, Gaussian(); data = dat, control = DRMControl(multi_start = 5))
control_info(fit5).start_logliks     # one log-likelihood per start
control_info(fit5).starts_agree      # false (and a warning) if they disagree
```

`maxit`, `start` and `multi_start` work on the fixed-effect routes of the
univariate families. You can request `:lbfgs` or `:bfgs` directly on those
primary ML fits. Hold the model, data and internal-scale start fixed when
comparing methods:

```julia
base = drm(f, Gaussian(); data = dat)
x0 = coef(base)
lb = drm(f, Gaussian(); data = dat,
         control = DRMControl(optimizer = :lbfgs, start = x0))
bf = drm(f, Gaussian(); data = dat,
         control = DRMControl(optimizer = :bfgs, start = x0))
control_info(bf).requested_optimizer   # :bfgs
control_info(bf).actual_optimizer      # :bfgs
control_info(bf).optimizer_runs        # every attempted start and native stopping result
```

`optimizer = :default` retains the route's current method and restart policy.
Gaussian ML with one mean random intercept `(1 | g)` also accepts `:lbfgs` or
`:bfgs`; its data-driven start, conditional restart, exact-gradient check and
boundary tie rule still apply. Its own two starts are reported separately in
`optimizer_runs`, and `selected_optimizer_stage` names the one retained.
`maxit`, `start`, `multi_start`, `seed` and `jitter` are not available on this random-intercept
route. Gaussian selection requires the ordinary ML route with
`algorithm = :auto`, and refuses alternate integration, imputation or
profiling options. Other random, structured, bivariate and REML routes refuse
an explicit optimizer by name. `profile_result`, `confint(...; method = :profile)`
and fit-based `bootstrap_result`, `bootstrap_summary` and `bootstrap_ci` refuse
fits with an explicit optimizer choice before starting any refit; they cannot
yet forward the selected method. Refit with `optimizer = :default` to use those
inference routes. Wald intervals remain available when standard errors were
computed. Solver agreement alone does not establish
parameter identification.

`se = false` works on every route:

```julia
fit = drm(f, Gaussian(); data = dat, control = DRMControl(se = false))
H = hessian(fit)                    # observed information, `coef(fit)` order
inv(H)                              # the covariance, computed later
g = gradient(fit)                   # ≈ 0 at the optimum
g' * (H \ g)                        # scaled Newton decrement: a stationarity audit
```

`gradient` and `hessian` are of the negative log-likelihood the
optimiser minimised, in the same order and on the same scale as `coef(fit)`.

## Step 3 — scale and standardise

Most convergence trouble is conditioning. Two cheap fixes:

- **Standardise covariates.** Predictors on wildly different scales make the
  Hessian ill-conditioned. Centre and scale continuous covariates before fitting.
- **Use ML for model comparison.** ML is the default and is comparable across
  fixed-effect structures; REML likelihoods are not. Keep ML when selecting
  models with [`aic`](@ref) / [`bic`](@ref).

## Comparing logLik with drmTMB or glmmTMB (#943)

For a `sigma ~ x + (1 | g)` model on `NegBinomial2()`, `Gamma()` or `Beta()`, the
default `:LA` route integrates the random intercept by adaptive Gauss-Hermite
quadrature. drmTMB and glmmTMB use the TMB Laplace approximation, so the two
log-likelihoods are **not comparable**: do not compare `loglik`, `aic` or an LRT
across the engines for these models. The estimates agree closely; the gap is the
integration convention, not a different optimum. Measured on lme4 `grouseticks`,
NB2 `ticks ~ year + height100 + (1 | brood)`:

| `sigma` formula | DRModels.jl default `:LA` | drmTMB / glmmTMB (TMB Laplace) |
|---|---|---|
| `sigma ~ year` | -888.350 | -888.885 |
| `sigma ~ 1` | -889.612 | -890.203 |

With `sigma ~ 1`, `marginal = :Laplace` reproduces the drmTMB value
(-890.2035). That route does not accept a non-constant `sigma`, and a
TMB-convention Laplace for `sigma ~ x + (1 | g)` is not planned for now, so
the comparison is not available for those models.

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
