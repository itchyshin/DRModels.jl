# Changing residual coupling with rho12

!!! note "Status — Stable"
    Mirrors drmTMB's [Changing residual coupling with rho12](https://itchyshin.github.io/drmTMB/articles/bivariate-coscale.html).
    **In DRModels.jl today:** bivariate Gaussian location–scale with a
    predictor-dependent residual correlation `ρ12` (fixed effects, ML).

With two responses, the interesting structure is often the **residual
correlation** ρ12 — how `y1` and `y2` co-vary *after* accounting for their means.
DRModels.jl lets ρ12 depend on predictors, with its own formula, exactly as drmTMB.

## A correlation that changes with a covariate

We simulate two standard responses whose residual correlation rises with `x`,
then recover that structure. ρ12 is modelled on the `atanh` scale (so it always
stays in `(-1, 1)`):

```@example bc
using DRModels, Random
Random.seed!(11)

n = 3000
x = randn(n)
ρ = tanh.(0.2 .+ 0.6 .* x)          # true residual correlation, rising with x
z1 = randn(n); z2 = randn(n)
y1 = z1
y2 = ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2
dat = (; y1, y2, x)

fit = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
             sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
             rho12 = @formula(rho12 ~ x)), Gaussian(); data = dat)

coef(fit, :rho12)        # atanh(ρ12): (Intercept), x  — ≈ [0.2, 0.6]
```

The coefficients are on the `atanh` scale. Back on the correlation scale, the
residual correlation at `x = 0` is:

```@example bc
tanh(coef(fit, :rho12)[1])     # ρ12 at x = 0  (≈ tanh(0.2) ≈ 0.197)
```

and it increases with `x` (positive `atanh` slope). The means (`mu1`, `mu2`) and
the scales (`sigma1`, `sigma2`) each have their own formula too — here the
scales were held constant (`~ 1`).

!!! note "Group-level vs residual correlation"
    `rho12` is the **residual** coupling of the two responses. Correlations that
    come from a shared phylogeny / spatial field / study are *group-level*
    covariance summaries, reported separately — not `rho12`.

## Phylogenetic location-scale coevolution

For the q=4 phylogenetic location-scale model, put the same
`phylo(1 | species)` marker on all four location/scale predictors: `mu1`,
`mu2`, `sigma1`, and `sigma2`. The residual `rho12` formula stays separate.

```julia
using DRModels, Random
Random.seed!(42)

phy = random_balanced_tree(6; branch_length = 0.2)
species = repeat(phy.leaf_names, inner = 3)
n = length(species)
x = randn(n)

# Small runnable example. Use larger, replicated simulations to study recovery.
u = Dict(name => 0.15 .* randn(4) for name in phy.leaf_names)
y1 = [1 + 0.4*x[i] + u[species[i]][1] +
      exp(-0.4 + u[species[i]][3]) * randn() for i in 1:n]
y2 = [-0.2 + 0.3*x[i] + u[species[i]][2] +
      exp(-0.5 + u[species[i]][4]) * randn() for i in 1:n]
dat = (; y1, y2, x, species)

fit_phy = drm(
    bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
       mu2 = @formula(y2 ~ x + phylo(1 | species)),
       sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
       sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
       rho12 = @formula(rho12 ~ 1)),
    Gaussian();
    data = dat,
    tree = phy,
    q4_vcov = false,
)

fit_phy.ranef.Sigma_a      # 4x4 group-level covariance, axes below
fit_phy.ranef.axes         # (:mu1, :mu2, :sigma1, :sigma2)
```

The `:phylocov` coefficient block describes group-level covariance rather than
a distributional predictor, so
[`predict_parameters`](@ref) returns `:mu1`, `:mu2`, `:sigma1`, `:sigma2`, and
`:rho12`, but not `:phylocov`. Use [`coevolution_cor`](@ref) for the among-axis
correlation matrix of `Σ_a`.

## Relmat / animal / spatial q=4 coevolution

The same q=4 model accepts level-indexed structured providers. Put
`relmat(1 | id)`, `animal(1 | id)`, or `spatial(1 | site)` on **all four** axes
and pass `K=…`, `A=…`, or `coords=…` respectively. Spatial uses a **fixed**
range (`spatial_range`; default = mean pairwise site distance), rather than
estimating range jointly. `bootstrap_sigma_a` is available only for tree-based
phylogenetic fits; calling it for the other structures raises an `ArgumentError`.

```julia
using DRModels, LinearAlgebra, Random
Random.seed!(189)

G = 8; nrep = 3
id = repeat([Symbol("g$k") for k in 1:G], inner = nrep)
n = length(id); x = randn(n)
R = randn(G, G); K = Matrix(Symmetric(R' * R + I))   # SPD relatedness
y1 = randn(n); y2 = randn(n)
dat = (; y1, y2, x, id)

fit_k = drm(
    bf(mu1 = @formula(y1 ~ x + relmat(1 | id)),
       mu2 = @formula(y2 ~ x + relmat(1 | id)),
       sigma1 = @formula(sigma1 ~ 1 + relmat(1 | id)),
       sigma2 = @formula(sigma2 ~ 1 + relmat(1 | id)),
       rho12 = @formula(rho12 ~ 1)),
    Gaussian(); data = dat, K = K, q4_vcov = false,
)
fit_k.ranef.Sigma_a
coevolution_cor(fit_k)
```

## See also

- [When variance carries signal](location-scale.md) — the single-response
  location–scale model.
- [Phylogenetic structured effects](phylogenetic-models.md) — tree input,
  interpretation, and limitations for phylogenetic models.
