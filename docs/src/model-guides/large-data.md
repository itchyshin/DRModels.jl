# Working with large data

!!! note "Status — Stable"
    Mirrors drmTMB's [Working with large data](https://itchyshin.github.io/drmTMB/articles/large-data.html). How DRModels.jl stays fast as the number of units grows, and what to reach for when a model is large.

The selling-point model — the q=4 phylogenetic bivariate location–scale fit — is
built to scale. The marginal likelihood is a **sparse augmented-state Laplace
approximation with an exact O(p) gradient**: it never forms the dense p×p
phylogenetic covariance, and it gets the gradient from a Takahashi selected
inverse rather than by differentiating a dense factorisation. The practical
consequence is near-linear scaling in the number of tips.

## What the timing evidence shows

Two synthetic timing studies define the current evidence boundary. In a
single-thread Julia sweep with four observations per tip, balanced and
caterpillar trees at `p = 100`, `1,000`, and `10,000` gave empirical wall-time
exponents of 0.90 and 0.91. That is evidence of near-linear growth for those
tree shapes and settings, not a runtime guarantee for a new dataset.

A separate paired benchmark used a synthetic near-balanced ultrametric tree,
four observations per tip, one discarded warm-up, two timed repetitions,
single-threaded linear algebra, and drmTMB 0.6.0 on the Totoro server:

| Tips | DRModels.jl median | drmTMB median | drmTMB / DRModels.jl |
|---:|---:|---:|---:|
| 100 | 0.529 s | 1.752 s | 3.31 |
| 1,000 | 6.921 s | 5.581 s | 0.81 |
| 5,000 | 82.141 s | 41.996 s | 0.51 |
| 10,000 | 115.178 s | 104.869 s | 0.91 |

DRModels.jl was faster at 100 tips in this comparison; drmTMB was comparable or
faster at the larger sizes. These figures describe one machine, package version,
tree shape, and model. They are not a general speed ratio, and the older
extrapolated “N× faster” claim is retired.

## Why it scales

- **Sparse precision, never dense covariance.** The phylogenetic prior precision
  is sparse (3N − 2 stored non-zeros for a tree with N nodes, about 6p for
  a binary tree with p tips). The engine factorises that sparse
  matrix with CHOLMOD; it never materialises the dense Σ.
- **Exact O(p) gradient.** The implicit-function gradient reuses a Takahashi
  selected inverse — the entries of the inverse that the sparse Cholesky already
  touches — instead of an O(p²) or O(p³) dense differentiation. This is the
  difference that keeps the iteration cost near-linear in the number of tips.
- **A precision sampler for uncertainty.** A latent-state draw uses the same
  sparse precision (`Cov(û) ≈ P⁻¹`) rather than a dense covariance. Total
  bootstrap time still depends on the number of refits and their convergence.

## Practical tips for large fits

- **Stay in ML.** ML is the default and is comparable across fixed-effect
  structures — keep it for model selection on large data. REML is an option, not
  the default.
- **Standardise covariates.** Good conditioning matters more as `p` grows; centre
  and scale continuous predictors so the optimiser's Hessian stays well-behaved.
- **Thread the bootstrap and the profile CIs.** Parametric bootstrap replicates
  are independent refits; profile-likelihood endpoints are independent per
  coefficient. `confint(fit; method = :profile, threads = true)` profiles
  coefficients in parallel when the objective is thread-safe — set
  `JULIA_NUM_THREADS` to engage it. Within a single coefficient the lower and
  upper endpoint chains stay serial, so the gain scales with the number of
  coefficients profiled, not with threads per coefficient.
- **Check the fit cheaply.** [`check_drm`](@ref) reports convergence and
  covariance conditioning without re-fitting — useful before committing to an
  expensive bootstrap.

## Beyond the verified engine

The O(p) machinery lives in the verified phylogenetic engine. The non-Gaussian
GLMM paths (Poisson/NB2/Beta/Gamma random effects via quadrature) are designed
for moderate group counts rather than p = 10,000-scale phylogenies; for very
large structured problems, the phylogenetic location–scale engine is the path
that has been benchmarked to scale.
