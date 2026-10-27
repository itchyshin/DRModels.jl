# DRModels.jl

[![Build Status](https://github.com/itchyshin/DRModels.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/itchyshin/DRModels.jl/actions/workflows/CI.yml)

DRModels.jl fits **distributional regression** models in Julia. In an ordinary
regression (`lm`, `lmer`, `glmmTMB`) a predictor changes only the *average*
response. In a distributional regression the average **and** the variability
each get their own formula, so you can ask whether, say, temperature changes
mean body size, how much individuals differ, or both.

Each parameter of the response distribution has a Greek letter in the
statistics and a keyword in the code:

| Parameter | Symbol | Keyword in formulas and `coef(fit, …)` |
|---|---|---|
| mean (location) | μ | `mu` (the response formula, `y ~ …`) |
| residual standard deviation (scale) | σ | `sigma` |
| residual correlation between two responses | ρ₁₂ | `rho12` |

If you know glmmTMB, the `sigma` formula plays the role of `dispformula`.

DRModels.jl is a standalone Julia package; no R installation is needed. It is
the Julia companion of the R package
[drmTMB](https://github.com/itchyshin/drmTMB) and uses the same model
vocabulary (`bf(...)`, `sigma`, `rho12`, `meta_V`). Documentation:
<https://itchyshin.github.io/DRModels.jl/dev/> (built from the current `main`
branch).

> **Status: experimental, version 0.7.2.** Install it from GitHub (it is not
> yet in the Julia General registry). The public API may still change across
> the 0.x series. Before choosing a model, check
> [what is tested](https://itchyshin.github.io/DRModels.jl/dev/capabilities).

## Install

You need **Julia 1.10 or later** ([download Julia](https://julialang.org/downloads/)).
Then, at the Julia prompt:

```julia
using Pkg
Pkg.add(url = "https://github.com/itchyshin/DRModels.jl")
using DRModels
```

Some documentation examples also use `Distributions` (to simulate data) or
`CairoMakie` and `AlgebraOfGraphics` (for figures). Add them the same way if a
page uses them, for example `Pkg.add("Distributions")`.

## A first model: average and variability

The example tells the same ecology story as drmTMB's
[first-model article](https://itchyshin.github.io/drmTMB/articles/drmTMB.html):
growth of 120 individuals sampled in forest and grassland, where habitat and
temperature shift mean growth and habitat also changes residual variation.
The true values and `n` match the R article. The printed estimates will not
match it, because Julia and R generate different random numbers even from the
same seed.

```julia
using DRModels, Random
Random.seed!(1)

n = 120
habitat = repeat(["forest", "grassland"]; inner = n ÷ 2)
temperature = randn(n)                 # n draws from a standard normal
grass = habitat .== "grassland"        # a dot (.==, .+, .*, exp.) applies an
                                       # operation to every element, like R's
                                       # vectorised arithmetic
# true values: mean = 1 + 0.6*grassland + 0.4*temperature
#              log(sigma) = -0.5 + 0.45*grassland
growth = (1 .+ 0.6 .* grass .+ 0.4 .* temperature) .+
         exp.(-0.5 .+ 0.45 .* grass) .* randn(n)

fit = drm(bf(@formula(growth ~ 1 + habitat + temperature),   # mean (mu)
             @formula(sigma ~ 1 + habitat)),                 # log residual SD
          Gaussian(); data = (; growth, habitat, temperature))

coef(fit, :mu)      # true values: [1.00, 0.60, 0.40]
coef(fit, :sigma)   # true values: [-0.50, 0.45]
coeftable(fit)      # Wald SEs, z, p, 95% CIs for every coefficient
```

`@formula(...)` writes a model formula, as `~` does in R.
`(; growth, habitat, temperature)` bundles the three vectors into a named
table. A `DataFrame` (for example one read from a CSV file with CSV.jl) works
the same way: `data = df`.

In R, the same model is
`drmTMB(bf(growth ~ habitat + temperature, sigma ~ habitat), family = gaussian(), data = d)`,
or `glmmTMB(growth ~ habitat + temperature, dispformula = ~ habitat, data = d)`.
For a Gaussian response, [glmmTMB 1.1.9 and later](https://glmmtmb.github.io/glmmTMB/news/index.html)
uses the same log-SD scale for its dispersion coefficients as `sigma` here.

```
──────────────────────────────────────────────────────────────────────────────────────
                            Estimate  Std.Error      z  Pr(>|z|)  Lower 95%  Upper 95%
──────────────────────────────────────────────────────────────────────────────────────
mu: (Intercept)             1.03555   0.0771317  13.43    <1e-40   0.884375   1.18673
mu: habitat: grassland      0.70105   0.150148    4.67    <1e-05   0.406766   0.995334
mu: temperature             0.327938  0.0673747   4.87    <1e-05   0.195886   0.45999
sigma: (Intercept)         -0.525039  0.091334   -5.75    <1e-08  -0.704051  -0.346028
sigma: habitat: grassland   0.503322  0.129232    3.89    <1e-04   0.250032   0.756612
──────────────────────────────────────────────────────────────────────────────────────
```

In this simulated fit, the displayed 95% intervals include their generating
values. Assessing interval coverage requires repeated simulations. The `sigma`
habitat coefficient is a
log residual-SD contrast: `exp(0.503) ≈ 1.65`, so residual SD in grassland is
about 1.65 times that in forest (true ratio `exp(0.45) ≈ 1.57`).

Next: the [Getting started](https://itchyshin.github.io/DRModels.jl/dev/getting-started) page explains each
step, how to read intervals, and what to check before you report a fit.

## What you can fit

The same `bf(...)` grammar covers:

- **15 response families**: Gaussian, Student-t, SkewNormal, Poisson,
  NegBinomial2, TruncatedNegBinomial2, TruncatedPoisson, Beta, BetaBinomial,
  Binomial, Gamma, LogNormal, ZeroOneBeta, Tweedie and CumulativeLogit
  (ordinal), plus zero-inflation (`zi`) and hurdle (`hu`) parts for counts.
- **Random effects** on the mean, such as `(1 | site)`, random slopes, and
  crossed or nested groups; for Gaussian models also on the scale,
  `sigma ~ 1 + (1 | site)`.
- **Structured random effects** from a known relatedness matrix (`relmat`), a
  pedigree (`animal`), a phylogeny (`phylo`) or coordinates (`spatial`).
- **Meta-analysis** with known sampling variances (`gaussian()` + `meta_V`).
- **Two responses at once** with a residual correlation `rho12`, including a
  bivariate phylogenetic location-scale model.

Support differs by family and structure. The
[capabilities page](https://itchyshin.github.io/DRModels.jl/dev/capabilities) lists, row by row, what is tested
and what is refused. One limit to know: Gaussian models with **crossed**
random intercepts are now fast at moderate sizes (about 0.3 s at 2,000 rows
and 45 s at 20,000 rows on a synthetic design of that shape,
[#977](https://github.com/itchyshin/DRModels.jl/issues/977)), but they are
still slower than drmTMB or glmmTMB at tens of thousands of rows.

**Inference.** When a usable covariance estimate is available, fits report
Wald standard errors and intervals. Check convergence and warnings before
using them.
Profile-likelihood intervals, parametric bootstrap intervals, likelihood-ratio
tests (`lrtest`), and AIC/BIC are available. Maximum likelihood (ML) is the
default; restricted ML (`method = :REML`) is available for a listed set of
Gaussian models. An interval being available is not a coverage guarantee: no
route claims calibrated coverage. See the
[capabilities page](https://itchyshin.github.io/DRModels.jl/dev/capabilities#inference) for the exact scope.

**From R.** An optional `engine = "julia"` route in drmTMB can call
DRModels.jl from R. That route is experimental; see
[Coming from R](https://itchyshin.github.io/DRModels.jl/dev/coming-from-r).

## Speed on a large phylogenetic model

The package's engine was first built for one demanding model: two traits
measured across `p` species, where a phylogenetic random effect is shared by
four linear predictors: the two means and the two log residual SDs
(written q = 4). Because a random effect enters the SD non-linearly, the
likelihood has no closed form and is approximated with a **Laplace
approximation** (the method glmmTMB and drmTMB also use). The retained
`q4_p100` comparison used the same data and Laplace ML objective in the two
packages. Its recorded results are shown below; timings depend on hardware,
package versions, and optimizer settings.

| | drmTMB | DRModels.jl |
|---|---|---|
| one fit, 100 species | 2.48 s, optimiser reported false convergence | **1.14 s, converged** |
| recorded log-likelihood | −256.52 | −256.51 |
| Wald SEs when a variance is estimated at zero | all `NaN` (Hessian not positive definite) | **finite for 16 of 17 parameters** |

The R run reported false convergence, which limits interpretation of the
timing ratio. Finite standard errors at a variance boundary do not establish
reliable interval coverage.

Separate synthetic scaling results in the retained grid reached 10,000
species for both packages. Under that protocol, drmTMB 0.6.0 was comparable
or faster at 1,000 species and above. These historical results concern the
recorded fixtures and environments. Full grid and caveats:
[report/comparison-grid.md](report/comparison-grid.md); per-family timings:
[report/speed-per-family.md](report/speed-per-family.md).

Run the benchmark fixtures from a clone of this repository:

```bash
julia --project=. bench/run_sparse_tmb_nd.jl     # retained q4_p100 fixture
julia --project=. bench/run_scaling.jl           # time vs number of species, to p=10,000
```

## Choosing an optimizer

For supported fixed-effect maximum-likelihood fits, you can request either
`:bfgs` or `:lbfgs`. Here `dat` is a data frame or named tuple with numeric
columns `y` and `x`; `:default` keeps the route's current optimizer and restart
policy.

```julia
using DRModels

form = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
fit_bfgs = drm(form, Gaussian(); data = dat,
               control = DRMControl(optimizer = :bfgs))
fit_lbfgs = drm(form, Gaussian(); data = dat,
                control = DRMControl(optimizer = :lbfgs))

control_info(fit_bfgs).requested_optimizer  # :bfgs
control_info(fit_bfgs).actual_optimizer     # optimizer used by the fit
```

Optimizer selection is limited to the primary ML routes described in the
[convergence guide](docs/src/model-guides/convergence.md), which also documents
restarts, unsupported routes, and inference refits.

## Engine details

For readers who want the numerical method behind the speed numbers above. The
bivariate phylogenetic location-scale model (PLSM) is fitted on a sparse
"augmented-state" precision matrix (`kron(Q_topology, Λ⁻¹)`), whose number of
non-zero entries grows linearly with the number of species `p` (written O(p)).
The gradient uses Takahashi selected inversion to obtain the needed inverse
entries without forming the dense p×p phylogenetic covariance. Sparse
factorization cost depends on the graph and fill. A fast mode-finder is tried first, and a more
robust one takes over when it fails.

## For contributors

- [HANDOVER.md](HANDOVER.md) records engineering provenance,
  [ROADMAP.md](ROADMAP.md) the development plan, and [AGENTS.md](AGENTS.md)
  the review roles and the Definition of Done.
- `src/` is the engine and public API; `src/experimental/` holds unwired
  prototypes and recorded negative results (per-file verdicts in
  [src/experimental/README.md](src/experimental/README.md)); `bench/` holds
  the runnable benchmarks and fixtures; `test/` the test suite; `report/` the
  design, provenance and benchmark reports; `docs/` the Documenter site
  (DocumenterVitepress) and its [CONTRACT.md](docs/CONTRACT.md).
- **Versions and registration.** The `v0.7.1` git tag predates the rename
  from `DRM` to `DRModels` and names the old package, so it is not a DRModels
  release tag. Registration in the Julia General registry is deferred until
  both DRModels.jl and drmTMB are ready; until then install via
  `Pkg.add(url = ...)`.
- The drmTMB comparison tests run only when `DRM_PARITY_TESTS=1` is set
  (RCall against fixtures generated with drmTMB v0.7.0); families are
  otherwise checked by simulation-based parameter recovery.
- To develop from a local clone
  (`git clone https://github.com/itchyshin/DRModels.jl`), use
  `Pkg.develop(path = "/absolute/path/to/DRModels.jl")` instead of `Pkg.add`.

## License

MIT © 2026 Shinichi Nakagawa. A sister package to drmTMB and
[GLLVModels.jl](https://itchyshin.github.io/GLLVModels.jl) (models of many
responses together).
