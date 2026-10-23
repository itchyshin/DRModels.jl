# Testing likelihoods

!!! note "Status — Implemented"
    Mirrors drmTMB's [Testing likelihoods](https://itchyshin.github.io/drmTMB/articles/testing-likelihoods.html). This page describes how DRModels.jl validates each likelihood today, through regular package checks and the engine-quality battery.

Every family and every random-effect path in DRModels.jl is checked the same way:
**write the likelihood, then prove it numerically** before it is wired into the
public API. There are four standing gates.

## 1. Parameter recovery — the primary gate

Each family is checked by **simulating from known coefficients, fitting, and
asserting recovery** within tolerance. This is the
Definition-of-Done gate: it confirms the log-likelihood, the link functions, and
the `sigma ↔ φ` mapping are all correct end to end. For example, the Gamma family
checks that the log-mean coefficients come back and that the shape recovers
through `α = 1/σ²`:

```julia
fit = drm(bf(y ~ x, sigma ~ 1), Gamma(); data = data)
@test coef(fit, :mu)[2] ≈ β_slope atol = 0.06
α̂ = exp(-2 * coef(fit, :sigma)[1])      # α = 1/σ²
@test α̂ ≈ α_true atol = 2.0
```

Recovery tests exist for the full surface: the Gaussian location–scale and
bivariate `ρ12` models, every non-Gaussian family, and each random-effect path
(`(1|g)`, `(1+x|g)`, crossed `(1|g)+(1|h)`, the `sigma`-RE GHQ marginal, and the
structured markers).

## 2. Exact-gradient vs finite differences

The verified engine's selling point is an **exact O(p) gradient**
(implicit-function / Takahashi selected inverse). It is held to a finite-difference
check: the analytic gradient must match a central finite difference to **≤ 1e-6**.
The same FD check guards the crossed sparse-Laplace family gradients (the nuisance
derivatives for NB2 / Gamma / Beta). A likelihood whose gradient fails FD does not
ship.

## 3. Collapse / consistency cross-checks

Where a general path should reduce to a simpler, independently-trusted one, the
tests assert it does:

- a **crossed** random-effect fit must match the **single-factor GHQ** path when
  one grouping factor collapses to a single level;
- **profile** confidence intervals must agree with **Wald** intervals on the
  well-identified mean coefficients (`confint(; method = :profile)` ≈
  `confint(; method = :wald)` on μ), while still bracketing the point estimate;
- in-sample `predict` must equal `fitted`.

These catch sign errors and mis-wired blocks that a single recovery test can miss.

## 4. AD-safety and the information matrix

Because the objective is written to be ForwardDiff-safe (`zero(eltype(θ))`, stable
log-densities, clamped predictors), the **Hessian** gives the observed-information
`vcov`, and the stored objective lets `confint(; method = :profile)` re-optimise
the nuisance parameters by inverting the likelihood ratio. The boundary-aware Wald
path maps a non-positive / non-finite stored variance to `Inf` (an unbounded CI
for an unidentified direction) rather than a silent `NaN`.

## Where the results live

The project keeps reproducibility records for its verification runs. Before a
release, the engine is checked for an FD-gradient error no larger than 1e-6, a
zero-allocation inner loop, and behaviour across several problem shapes. The
bar is **verify before claiming**: every speed or accuracy number in this
repository was reproduced by an independent run.
