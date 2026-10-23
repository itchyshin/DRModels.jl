# Known-matrix relatedness with relmat

!!! note "Status — Stable (Gaussian mean, supplied K)"
    Mirrors drmTMB's [Known-matrix relatedness with relmat](https://itchyshin.github.io/drmTMB/articles/relmat-known-matrices.html).
    **In DRModels.jl today:** `relmat(1 | id)` with a user-supplied relatedness matrix
    `K` on the Gaussian **mean** — a structured random intercept fit in closed
    form. `animal()` (pedigree) and `phylo()` (tree) reuse this engine.

When units are *related* by a known matrix — a pedigree, a phylogeny, a kinship
matrix — the random intercept is no longer i.i.d.: `u ~ N(0, σ_s² K)` with `K`
the known relatedness. For a Gaussian mean the marginal stays Gaussian,
`y ~ N(Xβ, D + σ_s² Z K Zᵀ)`, so DRModels.jl fits it in **closed form** (PGLS-style,
no approximation). Supply `K` (ordered by the grouping's first appearance):

```@example relmat
using DRModels, Random, LinearAlgebra
Random.seed!(1)

G = 40
A = let M = randn(G, G); M * M' / G + I end       # build a relatedness matrix
d = sqrt.(diag(A)); K = A ./ (d * d')              # → a correlation matrix
m = 5; n = G * m
id = repeat(1:G, inner = m)
x = randn(n)
u = 0.7 .* (cholesky(Symmetric(K)).L * randn(G))   # structured effect u ~ N(0, 0.7² K)
y = 0.3 .+ 0.5 .* x .+ u[id] .+ 0.4 .* randn(n)

fit = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
          Gaussian(); data = (; y, x, id), K = K)

re_sd(fit)[:id]        # structured-effect SD (≈ 0.7)
```

```@example relmat
exp(coef(fit, :sigma)[1])     # residual SD (≈ 0.4)
```

`K` must be ordered to match the levels of `id` as they first appear in the
data. The same engine powers `animal(1 | id)` (with a pedigree-derived `A`) and
`phylo(1 | species)` (with a tree-derived correlation).

!!! note "Gaussian mean vs location-scale"
    This short tutorial covers structured effects on the **mean**. If your
    scientific question is also about why the amount of between-group variation
    changes, use [location-scale-scale models](location-scale-scale.md). Those
    models use a different fitting method and need their own interpretation.

## See also

- [Which scale are you modelling?](../model-guides/which-scale.md) ·
  [What can I fit today?](../model-guides/model-map.md)
