# Coordinate-spatial structured effects

!!! note "Status — Stable (Gaussian mean, coordinate-direct)"
    Mirrors drmTMB's [Coordinate-spatial structured effects](https://itchyshin.github.io/drmTMB/articles/spatial-models.html).
    **In DRModels.jl today:** `spatial(1 | site)` on the Gaussian **mean** with site
    coordinates — an exponential spatial correlation `K(ρ) = exp(-d / ρ)` whose
    range `ρ` is estimated jointly. Closed-form GLS (coordinate-direct;
    mesh/SPDE is planned).

Nearby sites tend to be similar. `spatial(1 | site)` adds a random intercept
with a distance-based correlation built from the site coordinates,
`u ~ N(0, σ_s² K(ρ))`, and estimates the spatial **range** `ρ` along with the
spatial SD. Pass the coordinates via `coords =` (one row per `site` level):

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
          Gaussian(); data = (; y, x, site), coords = coords)

re_sd(fit)[:site]      # spatial SD (square it for the spatial variance)
```

`re_sd(fit)[:site]` is the spatial SD `σ_s`; `vc(fit)` is the matching
random-effect (co)variance summary for the structured block.

```@example sp
exp(coef(fit, :range)[1])     # estimated spatial range ρ
```

The spatial range is recovered only weakly from a single realization, so treat
its point estimate with care; the spatial **variance** and the fixed effects are
the robust outputs. `spatial`, `phylo`, `animal`, and `relmat` all share one
closed-form structured-GLS engine — only the source of the correlation differs
(coordinates / tree / pedigree / supplied matrix).

Check the fitted model before interpreting the spatial SD or range. Start with
[`check_drm`](@ref) and read its convergence, gradient, and Hessian messages;
then use [Improving convergence](../model-guides/convergence.md) if any of those
checks warn. The [Detailed capabilities & limits](../capabilities.md) page states
what this Gaussian spatial route does and does not establish about uncertainty.

!!! note "Gaussian only (for now)"
    A coordinate spatial effect is currently supported on the **Gaussian** mean.
    Routing `spatial(1 | site)` through the sparse-Laplace engine for the
    non-Gaussian families (Poisson, NB2, Gamma, Beta, Binomial) — alongside
    `relmat` / `animal` — is planned future work (`phylo` already has its
    non-Gaussian route).

## See also

- [Phylogenetic structured effects](phylogenetic-models.md) ·
  [Known-matrix relatedness with relmat](relmat-known-matrices.md)
- [What can I fit today?](../model-guides/model-map.md)
