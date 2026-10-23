# Structural dependence overview

!!! note "Status — Stable"
    Mirrors drmTMB's [Structural dependence overview](https://itchyshin.github.io/drmTMB/articles/structural-dependence.html). **In DRModels.jl today:** all four structured-effect markers on the mean — `relmat` (supplied `K`), `animal` (pedigree `A`), `phylo` (tree), and `spatial` (coordinates) — are available. For Gaussian responses with these effects on the mean, latent effects integrate out exactly; covariance parameters are estimated numerically. The q=4 phylogenetic bivariate engine is also available.

A structured effect puts a **known correlation** among the group-level random
intercepts instead of treating groups as independent. The marker names where
that correlation comes from:

| Marker | Correlation source | Supply |
|---|---|---|
| `relmat(1 \| id)` | an arbitrary known matrix `K` | `K = …` |
| `animal(1 \| id)` | an additive-genetic pedigree | `A = …` |
| `phylo(1 \| species)` | a phylogenetic tree | `tree = …` |
| `spatial(1 \| site)` | distance between coordinates | `coords = …` (range estimated) |

For Gaussian responses with Gaussian structured effects entering the mean
linearly and residual variance independent of those effects, all four have the
exact marginal covariance `V = D + σ_s² Z K Zᵀ`. No Laplace approximation is
needed in this case. Covariance parameters are still estimated numerically.
Each has its own worked tutorial — see below.

## See also

- [Known-matrix relatedness with relmat](relmat-known-matrices.md)
- [Animal models and additive relatedness](animal-models.md)
- [Phylogenetic structured effects](phylogenetic-models.md)
- [Coordinate-spatial structured effects](spatial-models.md)
