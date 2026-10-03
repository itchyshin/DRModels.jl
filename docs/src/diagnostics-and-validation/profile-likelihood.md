# Profile-likelihood intervals

!!! note "Status — Stable"
    Related drmTMB article: [First-week intervals: fit, profile, and boundary](https://itchyshin.github.io/drmTMB/articles/first-week-intervals.html); see also [`profile.drmTMB`](https://itchyshin.github.io/drmTMB/reference/profile.drmTMB.html).
    **In DRModels.jl today:** profile-likelihood confidence intervals via
    `confint(fit; method = :profile)`, the auditable [`profile_result`](@ref)
    object behind them, and the [`profile_curve`](@ref) data for a
    likelihood-ratio diagnostic plot.

A profile-likelihood interval asks a different question from a Wald interval. A
Wald interval uses the local curvature at the fitted estimate — fast, but only as
good as the quadratic approximation there. A **profile** interval fixes one
target at a sequence of values, re-optimises every other parameter at each one,
and keeps the values whose likelihood-ratio distance is still compatible with the
fitted model. When a parameter sits near a boundary or its likelihood is skewed,
the two intervals can differ noticeably, and a successfully computed profile can reveal asymmetry that Wald misses.

## Fit a small model

The example asks whether a continuous response changes with temperature while the
residual standard deviation is constant:

```math
y_i \mid \mu_i, \sigma \sim \operatorname{Normal}(\mu_i, \sigma^2),
\qquad \mu_i = \beta_0 + \beta_1\,\text{temperature}_i .
```

```@example prof
using DRModels, Random
Random.seed!(20260608)

n = 70
temperature = 2.4 .* rand(n) .- 1.2                 # Uniform(-1.2, 1.2)
growth = 0.5 .+ 0.8 .* temperature .+ 0.55 .* randn(n)
dat = (; growth, temperature)

fit = drm(bf(@formula(growth ~ temperature), @formula(sigma ~ 1)),
          Gaussian(); data = dat)
coef(fit, :mu)
```

## Wald first, then profile

`confint` returns one row per coefficient — `(param, coef, estimate, lower,
upper)` — on each parameter's working scale (μ on the response scale, σ on
`log σ`). The default is the Wald interval:

```@example prof
confint(fit)                          # method = :wald (the default)
```

Switch to the profile interval with `method = :profile`. Each endpoint is found
by re-optimising the nuisance parameters along a likelihood-ratio root-search, so
it costs more than Wald but does not assume a quadratic log-likelihood:

```@example prof
confint(fit; method = :profile)
```

For a single target, pass `parm` to profile just that parameter block — here the
constant residual scale, on the `log σ` scale:

```@example prof
confint(fit; method = :profile, parm = :sigma)
```

## The auditable profile object

`confint(...; method = :profile)` is a thin wrapper over
[`profile_result`](@ref), which returns the full object so a profile can be
**audited**, not just trusted. Alongside `ci` it reports the per-endpoint work
counts, the wall-clock time, and which nuisance-gradient backend was used:

```@example prof
res = profile_result(fit; parm = :sigma)
res.ci                                # same rows as confint(...; method = :profile)
```

```@example prof
(attempted = res.attempted, used = res.used, failed = res.failed,
 autodiff = res.autodiff, level = res.level)
```

The `stats` field carries one row per coefficient, with diagnostics for both
endpoints: likelihood evaluations, bracket expansions, endpoint failures and
searched-range limits. The generic profiler also records each arm's last
nuisance method, fallback use and acceptance reason:

```@example prof
res.stats
```

Check `lower_endpoint_failed` and `upper_endpoint_failed` first. A failed arm
has no valid endpoint: its signed infinity is a failure placeholder, not evidence
for an unbounded statistical interval. `confint` warns when an endpoint fails;
use `profile_result` to retain the diagnostics. The generic path rejects
non-finite or unsuccessfully terminated nuisance solves. Optimizer termination
alone does not prove a global optimum or a sufficiently small score.

If `lower_unbounded` or `upper_unbounded` is `true` without endpoint failure,
the threshold was not crossed **within the searched range**. This is a useful
diagnostic, not proof that the mathematical interval is infinite. Specialized
location-only and location-scale profilers retain their separate solver contracts;
the generic nuisance diagnostics do not certify those paths.

## Inspect the likelihood-ratio curve

To *see* the profile rather than only its endpoints, [`profile_curve`](@ref)
returns the data for a 1-D likelihood-ratio plot of one coefficient (by its
global index `k`). At each grid value of `θ[k]` the nuisance parameters are
re-optimised. Failed generic grid solves raise an error identifying the grid
location. A profiled objective materially below the reference fit also raises
an error instead of being clamped to zero. It returns the
grid `x`, the profile `deviance` (`2(ℓ̂ − ℓ_profile)`), the fitted `estimate`, and
the `χ²₁` `cutoff` the interval uses.

We profile the `log σ` coefficient. Its global index is the one whose
coefficient block is `:sigma`:

```@example prof
# global coefficient index of the (single) sigma coefficient
kσ = only(only(r for (p, r) in fit.blocks if p === :sigma))

curve = profile_curve(fit, kσ; npoints = 21)
(estimate = curve.estimate, cutoff = curve.cutoff, param = curve.param, coef = curve.coef)
```

The interval endpoints are where the `deviance` curve crosses `cutoff`. The
deviance is zero at the MLE and rises on both sides; the values of `x` where it
equals `cutoff` are exactly the profile endpoints reported above:

```@example prof
[(x = xi, deviance = di) for (xi, di) in zip(curve.x, curve.deviance)][1:5]
```

Crossings on both sides are useful checks, but do not by themselves certify
optimizer accuracy, interval coverage or the absence of other optima. Inspect
solver diagnostics and the curve. A nonmonotone curve requires investigation;
the numerical search does not guarantee finding its first crossing.

## When to reach for which

- **Wald** (`method = :wald`, the default) — fast, fine when the log-likelihood
  is near-quadratic around the estimate and the parameter is well inside its
  range.
- **Profile** (`method = :profile`) — slower but curvature-honest; prefer it for
  scale and correlation parameters, near boundaries, or whenever a Wald interval
  runs past a parameter's natural range.
- **Bootstrap** ([`bootstrap_ci`](@ref)) — when even the profile's asymptotics
  are in doubt; see the [figure gallery](figure-gallery.md) and the
  [post-fit tour](prediction-and-postfit.md).

## See also

- [Prediction, residuals & model comparison](prediction-and-postfit.md) — the post-fit accessor tour.
- [Did it converge?](../model-guides/convergence.md) — boundary and Hessian diagnostics with [`check_drm`](@ref).
- [Model fitting & post-fit reference](../reference/model-fitting-and-postfit.md) — every accessor's docstring.
