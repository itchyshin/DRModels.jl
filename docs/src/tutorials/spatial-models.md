# Coordinate-spatial structured effects

!!! note "Status: Stable (Gaussian mean, coordinate-direct)"
    Mirrors drmTMB's [Coordinate-spatial structured effects](https://itchyshin.github.io/drmTMB/articles/spatial-models.html).
    **In DRModels.jl today:** `spatial(1 | site)` on the Gaussian **mean** with site
    coordinates; an exponential spatial correlation `K = exp(-d / ρ) + 1e-6 I`
    whose range `ρ` is **fixed** at the median positive pairwise distance, as in
    drmTMB. Closed-form GLS, ML or REML (coordinate-direct; mesh/SPDE is planned).

!!! note "The range is fixed, not estimated"
    As in drmTMB, `spatial(1 | site)` does not estimate the range. It fixes `ρ`
    at the median positive pairwise distance between sites and estimates only the
    spatial SD. If the true range is far from the median distance the fixed
    kernel is a misspecified model: the spatial SD absorbs the difference. Before
    this change DRModels.jl estimated the range on the Gaussian route, which gave
    a different log-likelihood and spatial SD from drmTMB on the same data.

Nearby sites tend to be similar. `spatial(1 | site)` adds a random intercept
with a distance-based correlation built from the site coordinates,
`u ~ N(0, σ_s² K(ρ))`, with the spatial **range** `ρ` fixed at the median
positive pairwise distance; only the spatial SD is estimated. Pass one coordinate row per site and label the rows with `ids`.
The labels must match the values in the grouping column. A `Dict` from site
labels to coordinate vectors is another option. Unlabelled matrices retain
first-seen site ordering and emit a warning; their rows are not sorted by label.
Extra labelled sites are dropped, while a missing site label raises an error.
If the model also uses `animal()`, `ids` labels the pedigree matrix `A`; use a
`Dict` to label the spatial coordinates in that case.

```@example sp
using DRModels, Random, LinearAlgebra
Random.seed!(8)

G = 50
coords = rand(G, 2) .* 10.0                  # site locations in [0,10]²
Ddist = [sqrt(sum(abs2, coords[k, :] .- coords[l, :])) for k in 1:G, l in 1:G]
K = exp.(-Ddist ./ 2.0) + 1e-8 * I           # true correlation, range = 2
u = 0.9 .* (cholesky(Symmetric(K)).L * randn(G))

m = 4; n = G * m
site = repeat(1:G, inner = m)
x = randn(n)
y = 0.3 .+ 0.5 .* x .+ u[site] .+ 0.4 .* randn(n)

fit = drm(bf(@formula(y ~ x + spatial(1 | site)), @formula(sigma ~ 1)),
          Gaussian(); data = (; y, x, site), coords = coords, ids = collect(1:G))

re_sd(fit)[:site]      # spatial SD (square it for the spatial variance)
```

`re_sd(fit)[:site]` is the spatial SD `σ_s`; `vc(fit)` is the matching
random-effect (co)variance summary for the structured block.

```@example sp
using Statistics
median(Ddist[Ddist .> 0])     # the fixed range ρ (median positive pairwise distance)
```

This is one simulated realization; it does not establish general parameter
recovery or interval calibration. Inspect uncertainty for both the spatial SD
and range. For Gaussian mean models, `spatial`, `phylo`, `animal`, and `relmat`
share the closed-form structured GLS engine; the source of the correlation
differs (coordinates, tree, pedigree, or supplied matrix).

Check the fitted model before interpreting the spatial SD. Start with
[`check_drm`](@ref) and read its convergence, gradient, and Hessian messages;
then use [Improving convergence](../model-guides/convergence.md) if any of those
checks warn. The [Detailed capabilities & limits](../capabilities.md) page states
what this Gaussian spatial route does and does not establish about uncertainty.

!!! note "Spatial routes depend on the family"
    This tutorial uses the exact Gaussian coordinate-spatial route. Every route
    that takes `coords` fixes the range as drmTMB does: the kernel is `exp(-d / m) + 1e-6 I`, with
    `m` the median positive pairwise distance between sites.

    - **Gamma** accepts `spatial(1 | site)` with `coords` (native kernel) or
      with a supplied `K`, not both. A 1-D vector of coordinates is accepted.
    - **NB2** accepts `spatial(1 | site)` with a supplied `K`. With `coords` it
      accepts only a slope-only `spatial(0 + x | site)` term with its
      `sd(...)` formula.
    - **Beta** accepts `spatial(1 | site)` with a supplied `K` when `sigma ~ 1`
      and no SD formula is used. With `sd(site, spatial) ~ z` it takes
      `coords` (native kernel) and refuses `K`.
    - **Binomial** refuses a bare `spatial(1 | site)` on the mean. It accepts
      `spatial` with an `sd(site, spatial)` formula and `coords`, on the same fixed
      median-distance kernel as drmTMB (earlier versions used the mean pairwise
      distance and a 1e-8 ridge).

    Check the family-specific restrictions in
    [Detailed capabilities and limits](../capabilities.md).

## See also

- [Phylogenetic structured effects](phylogenetic-models.md) ·
  [Known-matrix relatedness with relmat](relmat-known-matrices.md)
- [What can I fit today?](../model-guides/model-map.md)
