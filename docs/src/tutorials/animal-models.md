# Animal models and additive relatedness

!!! note "Status — Stable (Gaussian mean, supplied A)"
    Mirrors drmTMB's [Animal models and additive relatedness](https://itchyshin.github.io/drmTMB/articles/animal-models.html).
    **In DRModels.jl today:** `animal(1 | id)` on the Gaussian **mean** with a supplied
    additive-relatedness matrix `A` — a structured random intercept fit in closed
    form (same engine as [`relmat`](relmat-known-matrices.md) and
    [`phylo`](phylogenetic-models.md)).

The quantitative-genetic *animal model* splits phenotypic variation into an
additive-genetic random effect with a known relatedness matrix `A` plus a
residual. Each individual `i` carries a breeding value `a_i`, and the breeding
values of related individuals covary in proportion to how much genetic material
they share — encoded by `A` (built from a pedigree). The model is

```math
y = X\beta + a[\mathrm{id}] + \varepsilon,\qquad
a \sim N(0,\ \sigma_A^2\,A),\qquad
\varepsilon \sim N(0,\ \sigma^2).
```

`σ_A²` is the *additive-genetic variance* and `σ²` the residual variance. Because
the breeding effect enters the **mean** linearly and is Gaussian, the marginal of
`y` stays Gaussian — `y ~ N(Xβ, σ² I + σ_A² Z A Zᵀ)` — so DRModels.jl fits it in
closed form (PGLS-style), the same engine that powers
[`phylo`](phylogenetic-models.md) and [`relmat`](relmat-known-matrices.md).

## The DRModels.jl formula

Write the breeding effect as `animal(1 | id)` in the mean formula and supply the
relatedness matrix with the `A =` keyword. `A` is a symmetric positive matrix
indexed over the levels of `id` (one row/column per individual):

```julia
drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)),
    Gaussian(); data = (; y, x, id), A = A)
```

`animal(1 | id)` and `relmat(1 | id)` are the same structured random intercept;
`animal` simply names the genetic interpretation. The companion
[`phylo(1 | species)`](phylogenetic-models.md) marker takes a `tree =` instead of
`A =` and builds the correlation from the phylogeny.

## A worked Gaussian example

Simulate a balanced design — `G` individuals, `m` records each — with a known
additive-relatedness matrix `A`, then recover the variance components:

```@example animal
using DRModels, Random, LinearAlgebra
Random.seed!(3)

G = 60
M = randn(G, G); A0 = M * M' / G + I
d = sqrt.(diag(A0)); A = A0 ./ (d * d')          # additive-relatedness matrix
m = 6; n = G * m
id = repeat(1:G, inner = m)
x = randn(n)
a = 0.8 .* (cholesky(Symmetric(A)).L * randn(G)) # additive-genetic effect, σ_A = 0.8
y = 0.3 .+ 0.5 .* x .+ a[id] .+ 0.4 .* randn(n)  # residual SD 0.4

fit = drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)),
          Gaussian(); data = (; y, x, id), A = A)

re_sd(fit)[:id]           # additive-genetic SD σ_A (≈ 0.8)
```

## Reading the fit

`re_sd`, `coef`/`sigma`, and `ranef` read off the pieces of an animal model.
The additive-genetic and residual SDs:

```@example animal
re_sd(fit)[:id]               # additive-genetic SD σ_A
```

```@example animal
exp(coef(fit, :sigma)[1])     # residual SD σ (≈ 0.4)
```

!!! note "Per-individual breeding values"
    `ranef` surfaces conditional breeding values (BLUPs) for the crossed /
    correlated Gaussian random-effect paths. For a single structured
    `animal()` / `relmat()` component they are estimated internally but not yet
    returned — `ranef(fit)` is empty here — so read the genetic signal off
    `re_sd` and the heritability below.

The **narrow-sense heritability** `h² = σ_A² / (σ_A² + σ²)` follows directly from
the two SDs:

```@example animal
σA = re_sd(fit)[:id]; σ = exp(coef(fit, :sigma)[1])
σA^2 / (σA^2 + σ^2)           # heritability h² (≈ 0.8² / (0.8² + 0.4²) = 0.8)
```

!!! tip "vc is for correlated blocks"
    `re_sd` reports the scalar random-intercept SD used by `animal(1 | id)`.
    `vc` returns a full random-effect *covariance matrix* and is meant for
    **correlated** blocks like `(1 + x | g)`; a scalar animal intercept has no
    covariance term to report.

## Scope and what's next

Today's animal-model path covers the **Gaussian mean** with a supplied `A`.
Building `A` from a pedigree and a sparse large-pedigree path are planned.
Non-Gaussian animal models (Poisson / NB2 / Gamma / Beta / Binomial breeding
effects routed through the sparse-Laplace GLMM engine) are planned future work.
The phylogenetic non-Gaussian route already exists and the `relmat`/`animal`
route is intended to reuse it, so don't assume non-Gaussian families work here
yet.

## See also

- [Known-matrix relatedness with relmat](relmat-known-matrices.md) ·
  [Phylogenetic structured effects](phylogenetic-models.md)
- [What can I fit today?](../model-guides/model-map.md)
