# Structural dependence details

!!! note "Status — Theory + roadmap"
    Mirrors drmTMB's [Structural dependence details](https://itchyshin.github.io/drmTMB/articles/phylogenetic-spatial.html). The theory of combining structured dependences. **In DRModels.jl today:** each structured effect — `phylo`, `spatial`, `animal`, `relmat` — is fitted on its own; a *simultaneous* phylogenetic × spatial fit is planned.

Ecological and evolutionary data often carry **more than one** source of
non-independence at once: closely related species share trait values
(phylogeny), and nearby sites share environments (space). This page explains how
those dependences are represented, what DRModels.jl fits today, and what a combined
model will add.

## One structured effect at a time

A structured random effect replaces the independent prior `u ~ N(0, σ² I)` with
a **known correlation matrix** `C`:

| Marker | Correlation `C` | Source |
|---|---|---|
| `phylo(1 \| species)` | phylogenetic (Brownian) | a tree |
| `spatial(1 \| site)` | `exp(-d / ρ)` | coordinates + estimated range `ρ` |
| `animal(1 \| id)` | additive-genetic | a pedigree `A` |
| `relmat(1 \| id)` | arbitrary supplied | a matrix `K` |

Each is a Gaussian random intercept `u ~ N(0, σ² C)` on the mean, fitted in
closed form (GLS). The group SD comes back from [`re_sd`](@ref). These are all
**Stable** today — see the [phylogenetic](phylogenetic-models.md) and
[spatial](spatial-models.md) tutorials.

```julia
using DRModels
# phylogenetic random intercept from a tree
fitp = drm(bf(@formula(y ~ x + phylo(1 | species))), Gaussian(); data = dat, tree = tr)

# spatial random intercept from coordinates (separate model)
fits = drm(bf(@formula(y ~ x + spatial(1 | site))), Gaussian(); data = dat, coords = xy)
```

## Combining dependences (planned)

A *simultaneous* phylogenetic-and-spatial model carries both random effects in
the same fit:

```math
y = X\beta + u_{\text{phylo}} + u_{\text{space}} + \varepsilon,
\qquad u_{\text{phylo}} \sim N(0, \sigma_p^2 C_p),
\qquad u_{\text{space}} \sim N(0, \sigma_s^2 C_s).
```

The marginal covariance is then `V = σ_p² Z_p C_p Z_pᵀ + σ_s² Z_s C_s Z_sᵀ + D`
— a sum of two structured blocks plus the residual. Identifying the two variance
components requires that the two correlation patterns are not collinear (related
species must not be perfectly co-located); when they are well separated, the data
distinguish "evolutionary" from "spatial" signal.

!!! note "Not fitted yet"
    DRModels.jl currently resolves a **single** structured marker per formula, so a
    `phylo(...) + spatial(...)` mean is on the roadmap rather than fitted today.
    The building blocks already exist — the closed-form GLS for each structure,
    and the multi-component Woodbury path used for crossed ordinary random
    effects (`(1 | g) + (1 | h)`) — so the combined fit is an extension of
    machinery that is already in place, not new theory.

## Why it matters

Fitting only one structure when both are present biases the estimated variance
of the one you keep (it absorbs the signal of the one you dropped) and can bias
the fixed effects if the omitted structure is correlated with a covariate. Until
the combined fit lands, the practical recommendation is to fit each structure
separately, compare the variance components and the fixed-effect estimates across
the two fits with `aic` / `bic`, and treat a large shift as a
warning that both dependences are active and a joint model is needed.
