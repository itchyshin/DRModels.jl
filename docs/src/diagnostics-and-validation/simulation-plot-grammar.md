# Simulation plot grammar

!!! note "Status — Implemented"
    Mirrors drmTMB's [Simulation plot grammar](https://itchyshin.github.io/drmTMB/articles/simulation-plot-grammar.html). DRModels.jl keeps the base package **plotting-dependency-free**: the helpers below return the *numbers a plot needs* — grids, deviances, correlations. Optional drawing is provided by **`DRModelsMakieExt`** (Makie + AlgebraOfGraphics weakdeps) via [`drm_figure`](../reference/visualization.md) / thin `plot_*` aliases. Default CI does **not** load Makie or render figures — only the method-less stub is gated.

There are two ingredients: **`simulate`** (generate replicate data from a fit) and
the **plot-data providers** (turn a fit into plottable grids).

## Simulating from a fit

`simulate(fit; rng)` draws a parametric replicate of the response from the fitted
model — the same machinery the parametric bootstrap (`bootstrap_ci`) uses:

```julia
y_rep = simulate(fit)            # one replicate on the response scale
```

Re-fitting many replicates and summarising the spread of the estimates is the
**recovery study** (ADEMP) pattern used throughout `test/` — simulate from known
coefficients, fit, and check the truth is recovered. That loop is exactly what a
simulation plot visualises.

## Plot-data providers

All three return a plain `NamedTuple` of numbers; you supply the backend.

### `profile_curve(fit, k; npoints = 41, span = 3.0, level = 0.95)`

The 1-D **profile-likelihood** curve for coefficient index `k`. At each grid value
of `θ[k]` the nuisance parameters are re-optimised, so it is a genuine
likelihood-ratio profile, not a quadratic/Wald approximation. Returns
`(x, deviance, estimate, cutoff, k, param, coef, level)`:

- `x` — grid spanning `θ̂[k] ± span·se[k]`, including the MLE exactly;
- `deviance` — `2(ℓ̂ − ℓ_profile)` at each `x`;
- `cutoff` — the `χ²₁(level)` reference line the profile interval inverts.

```julia
pc = profile_curve(fit, 2)
# plot(pc.x, pc.deviance); hline!([pc.cutoff]); vline!([pc.estimate])
```

### `parameter_surface(fit, k1, k2; npoints = 25, span = 3.0)`

The 2-D profile-deviance **surface** over two coefficients — the data behind
`plot_parameter_surface`. At each node the remaining parameters are profiled out,
so `z` is the true profile deviance, `0` at the MLE. Returns `(x, y, z, k1, k2)`
with `z` an `npoints × npoints` matrix; `χ²₂` contours give joint confidence
regions.

```julia
ps = parameter_surface(fit, 1, 2)
# contour(ps.x, ps.y, ps.z')
```

### `corpairs_data(fit)`

The estimated-coefficient **correlation summary** behind `plot_corpairs` — the
off-diagonal correlations of the coefficient covariance, ready for a pairs /
heatmap panel.

## The Confidence Eye

Florence's **Confidence Eye** contract: a pale compatibility region, a darker
outline, and a hollow point estimate. With Makie + AlgebraOfGraphics loaded,
`drm_figure(profile_curve(...); kind = :profile)` (or `plot_profile`) draws that
Eye on the profile interval. The Documenter figure gallery may also render Eyes
from `confint` rows — that is docs/gallery evidence, not default-CI evidence.

Because every provider returns numbers (not figures), the base package stays free
of a plotting dependency; drawing is opt-in via `DRModelsMakieExt`.
