# Adding distribution families

!!! note "Status — Developer guide"
    Mirrors drmTMB's [Adding distribution families](https://itchyshin.github.io/drmTMB/articles/adding-families.html). This page documents how families are actually built in DRModels.jl today (the [`Poisson`](@ref) family is the worked reference). Adding a family is **Workflow H** (the `add-family` skill); it is the once-per-family loop of Phase 2.

A family in DRModels.jl is a small **struct** plus a `drm` method that builds the design
matrices, writes the negative log-likelihood, and optimises it by maximum
likelihood. Everything else — the `bf` front end, the design builder, the
post-fit accessors, inference — is shared. Adding a family is therefore mostly
"write the log-likelihood and wire it in".

## The seven steps

### 1. Define the family struct

```julia
struct MyFamily end
```

Give it a docstring that states the **links** (which scale each parameter acts on)
and what the `sigma` slot means for this family. Keep the parameter names stable:
the scale is always `sigma` (never `tau`); for families whose natural nuisance is a
dispersion/shape, the `sigma` slot carries it on a **log link** — preserve the
documented `sigma ↔ φ` mapping (e.g. Beta/BetaBinomial `φ = 1/σ²`, Gamma `α = 1/σ²`).

### 2. Add a `drm` method

Dispatch on your family and reuse the shared helpers:

```julia
function drm(f::DrmFormula, fam::MyFamily; data, g_tol::Real = 1e-8)
    rhs = Dict(f.forms)
    fixed_mu, re, mv, st = _split_ranef(rhs[:mu])   # peel off (1|g), meta_V, structured markers
    (mv === nothing && st === nothing) ||
        error("MyFamily() does not support meta_V / structured markers")
    y, Xμ, nmμ = _design(f.response, fixed_mu, data)  # response, design matrix, coef names
    # ...validate the response domain (e.g. counts, (0,1), positive)...
    return _withformula(_fit_myfamily(fam, y, Xμ, nmμ, g_tol), f)
end
```

`_split_ranef` separates fixed effects from random-effect / marker terms;
`_design` applies the StatsModels schema (including R's implicit intercept) and
returns `(y, X, coefnames)`. Validate the response domain and error clearly for
anything not yet supported.

### 3. Write the fitter

The fitter defines a ForwardDiff-safe `nll(θ)`, optimises it, and assembles a
`DrmFit`. This is the core of the work:

```julia
function _fit_myfamily(fam::MyFamily, y, Xμ, nmμ, g_tol)
    n = length(y); pμ = size(Xμ, 2)
    function nll(θ)
        ημ = Xμ * θ                       # linear predictor (apply the link here)
        s = zero(eltype(θ))               # eltype(θ) — keeps it AD-safe
        @inbounds for i in 1:n
            s -= logpdf_myfamily(y[i], ημ[i])
        end
        return s
    end
    θ0 = zeros(pμ); θ0[1] = ...           # a sensible intercept start (e.g. log mean)
    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    θ̂ = Optim.minimizer(res); V = inv(ForwardDiff.hessian(nll, θ̂))
    blocks = [:mu => 1:pμ]; names = [:mu => nmμ]
    means = Dict(:mu => ...); obs = Dict(:mu => Vector{Float64}(y))   # response-scale fitted + observed
    scales = Dict{Symbol,Vector{Float64}}()
    return _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n, Optim.converged(res), means, obs, scales), nll)
end
```

Conventions that matter:

- **AD-safety** — write `s = zero(eltype(θ))` and prefer explicit, stable
  log-density expressions; `clamp` linear predictors (e.g. to `[-30, 30]`) where a
  constructor or `exp` could overflow.
- **`blocks` / `names`** — one `Symbol => UnitRange` per parameter block; the same
  symbols key `coef(fit, :mu)`. Use `:resd` / `:recov` for random-effect SD /
  covariance blocks (the convention the `re_sd` / `vc` accessors read).
- **`means` / `obs`** are stored on the response scale so `fitted`, `residuals`,
  `predict`, and `simulate` work for free.
- **`_withnll`** attaches the objective so `confint(; method = :profile)` and the
  parametric bootstrap work for free.

### 4. Wire it into the module

In `src/DRModels.jl`, `include("myfamily.jl")` next to the other families and add the
struct to the family export list.

### 5. Add a recovery test

Add `test/test_myfamily.jl` — simulate from known coefficients, fit, and assert
recovery within tolerance — then `include` it in `test/runtests.jl`. This is the
Definition-of-Done gate.

### 6. Document it

Add the struct to the [Model specification](@ref) reference `@docs` block and
write/extend a tutorial with a runnable `@example`.

### 7. Optional: random effects, zero-inflation, hurdle

The [`Poisson`](@ref) family shows the full menu, all routed from its `drm`
method:

- **`(1 | g)`** — integrate the random intercept out by **Gauss–Hermite
  quadrature** (`_gauss_hermite`), `b = √2 σ_b z`.
- **`(1 + x | g)`** — per-group adaptive Gauss–Hermite quadrature through the shared
  helper `_aghq_marginal_loglik` (`src/adaptive_ghq.jl`, #834): the family supplies
  only the per-observation log-density `ll(i, η)`; the grid is centred on each
  group's mode and scaled by its curvature (5 nodes per axis; 1 node = Laplace).
- **`(1 | g) + (1 | h)`** crossed — the shared sparse-Laplace spine
  (`sparse_laplace_glmm.jl`).
- **`zi` / `hu`** — a second linear predictor for the zero-inflation / hurdle
  probability (logit link), added as its own `bf` formula.

Start with the fixed-effects fitter; add these only once the base family recovers.

## Checklist

| Step | Artefact |
|---|---|
| 1 | `struct MyFamily end` + docstring (links, `sigma`↔φ) |
| 2 | `drm(::DrmFormula, ::MyFamily; …)` method |
| 3 | `_fit_myfamily` with AD-safe `nll`, `DrmFit`, `_withnll` |
| 4 | `include` + export in `src/DRModels.jl` |
| 5 | `test/test_myfamily.jl` recovery test + `runtests.jl` |
| 6 | reference `@docs` entry + tutorial `@example` |
| 7 | (optional) RE / `zi` / `hu` paths |
