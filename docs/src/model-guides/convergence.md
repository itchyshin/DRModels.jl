# Improving convergence

!!! note "Status — Stable"
    Mirrors drmTMB's [Improving convergence](https://itchyshin.github.io/drmTMB/articles/convergence.html). A practical guide to checking and improving a fit in **DRModels.jl**, built around the [`check_drm`](@ref) diagnostic.

The optimiser and stopping tests depend on the fitted route. Use this page
when a fit reports a numerical problem, and to distinguish an early stop from
a model sitting at the edge of its parameter space.

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

- `convergence_status` reports `:converged`, `:boundary`, or `:degenerate`.
  `converged` is `true` exactly when this status is `:converged`, matching
  [`is_converged`](@ref). The raw optimiser flag is `fit.converged`.
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
- `ok` combines the raw convergence/nondegeneracy check, a small available
  gradient, a positive-definite covariance, and no fixed-effect rank deficiency.
  The gradient condition is skipped when unavailable or for penalised MAP fits.
  This summary differs from `is_converged`: `ok` can be `true` at a variance
  boundary or when the status flags inflated standard errors. Read both fields.

!!! warning "Read the status as well as the raw flag"
    `fit.converged = true` says the optimiser's stopping rule was met. It does
    not prove an interior, identified maximum. `is_converged(fit)` additionally
    checks the public convergence status and returns `false` at `:boundary`
    or `:degenerate`, including detected variance/correlation boundaries and
    non-finite or inflated standard errors when uncertainty was computed.
    These checks are numerical diagnostics, not a model-adequacy test. Read
    `max_abs_grad`, `grad_source`, the covariance checks, and the fitted
    parameter values together.

## Step 2 — read the failure mode

A non-`ok` report is information, not necessarily an error.

### A variance at the boundary

If a random-effect SD or a scale parameter is estimated near zero, the
likelihood flattens in that direction: `min_eigval` is tiny and the gradient may
not reach zero. This can be the MLE when the data provide little information
about that variance component. Do not assume the other parameters' Wald
standard errors remain reliable; inspect their own identification and
covariance diagnostics. A profile interval avoids the quadratic Wald
approximation, but its likelihood-ratio calibration is still asymptotic and
boundary likelihood-ratio limits can be nonstandard [Self and Liang
(1987)](https://doi.org/10.1080/01621459.1987.10478472).

This advice applies to variance components estimated at zero. A Student-t
shape parameter near its infinite-`nu` Gaussian limit is a different
identification problem: the data may weakly distinguish large values of `nu`.
If its standard error is zero or non-finite, do not interpret the Wald
interval; the optimizer flag alone does not make that uncertainty reliable.
Compare with a Gaussian fit when it answers the same question, and use a
profile or bootstrap interval only when the target and evidence support it.

### Confounded predictors

A near-singular covariance with a large `cond` usually means two columns of a
design matrix are nearly collinear (e.g. an intercept and a centred covariate
that is almost constant). Re-centre or drop one; refit.

An *exactly* aliased column is a different case: `sigma ~ 1 + x + z` with
`z = 2x + 1`, or a duplicated column, leaves the likelihood flat along the alias
direction, so the coefficient is not identified at all. `drm` (Gaussian
location-scale route) checks the rank of each parameter's fixed-effect design
once, warns with `fixed_effect_rank_deficiency` naming the parameter block and the
aliased coefficients, and reports `is_converged(fit) == false`. The aliased
coefficients' standard errors are `NaN`, and `check_drm(fit).ok` is `false` with the
blocks listed in `check_drm(fit).fixed_effect_rank_deficiency`. The log-likelihood
and the identified coefficients are unchanged (the covariance is still the
pseudo-inverse); drop or recode the aliased column and refit. This matches
drmTMB's `convergence_status() == "degenerate"` for the same designs.

### Stopped early

If `converged` is `false` but `max_abs_grad` is still sizable, the optimiser ran
out of iterations rather than hitting a boundary. A smaller `g_tol` will not help:
it is the gradient size at which the optimiser may stop, so tightening it makes
stopping harder, not easier. To raise the iteration cap or supply starting
values, pass one `DRMControl` (the names follow drmTMB's `drm_control()`):

```julia
f = bf(@formula(y ~ x), @formula(sigma ~ x))
fit = drm(f, Gaussian(); data = dat, control = DRMControl(maxit = 2000))

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
bfgs_fit = drm(f, Gaussian(); data = dat,
         control = DRMControl(optimizer = :bfgs, start = x0))
control_info(bfgs_fit).requested_optimizer   # :bfgs
control_info(bfgs_fit).actual_optimizer      # :bfgs
control_info(bfgs_fit).optimizer_runs        # every attempted start and native stopping result
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

Check conditioning first. Two cheap fixes:

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
(sparse augmented-state Laplace with an exact sparse gradient). Two notes specific
to it:

- **Off-diagonal initialisation.** Starting the cross-covariance Cholesky away
  from zero (the `lc3`/`lc7` entries) avoids a saddle near the diagonal start.
- **The Watanabe-singular boundary.** When a per-axis phylogenetic variance is
  near zero the model sits at a singular point; the optimiser flags the plateau
  rather than chasing a flat ridge. This matches drmTMB's behaviour (it reports
  "false convergence") — both engines are correct; the geometry, not the
  optimiser, is the limiting factor.
