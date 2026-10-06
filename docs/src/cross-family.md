# Cross-family bivariate dependence

!!! note "Status — Experimental"
    Bivariate modelling for **two responses from different families** (for
    example, Gaussian × Poisson), using a shared per-observation latent.
    Fit through the `bf(...)` front end, for example
    `drm(bf(...), (Gaussian(), Poisson()); data = …)`. `Gaussian`, `Poisson`, `Binomial`,
    `NegBinomial2`, `Beta` and `Gamma` axes are supported; the route takes family *instances*, e.g.
    `(Gaussian(), Poisson())`. The dependence is reported on the link/latent
    scale; the residual-correlation [`rho12`](tutorials/bivariate-coscale.md)
    model is the Gaussian × Gaussian special case.

The residual-correlation model
([Changing residual coupling with rho12](tutorials/bivariate-coscale.md))
couples two **Gaussian** responses through a residual correlation `ρ12`. But
often the two responses are not from the same family — a continuous trait and a
count, a count and a proportion. There is no single residual covariance matrix
to write down, because the two responses live on different scales with different
mean–variance relationships. The cross-family `drm(...)` route handles that
case through a shared latent variable.

## The model

Each observation `i` carries a single shared scalar latent `uᵢ ~ N(0, 1)`. The
two linear predictors load on it with their own loadings `λ₁`, `λ₂`:

```math
\eta_{1i} = X_{1i}\,\beta_1 + \lambda_1\,u_i,
\qquad
\eta_{2i} = X_{2i}\,\beta_2 + \lambda_2\,u_i,
```

and each response is then drawn from **its own family**, conditional on its own
linear predictor:

```math
y_{1i} \sim \text{fam}_1(\eta_{1i}),
\qquad
y_{2i} \sim \text{fam}_2(\eta_{2i}).
```

The shared `uᵢ` is what induces the dependence: a large draw pushes both
`η₁ᵢ` and `η₂ᵢ` in directions set by the signs and sizes of `λ₁` and `λ₂`. With
`λ₁ λ₂ > 0` the two responses move together; with `λ₁ λ₂ < 0`, they oppose.

Because `uᵢ` is a single scalar per observation, the marginal likelihood is a
**one-dimensional** integral, evaluated by Gauss–Hermite quadrature with `K`
nodes:

```math
p(y_{1i}, y_{2i}) =
\int_{-\infty}^{\infty}
  p_1(y_{1i}\mid \eta_{1i})\,
  p_2(y_{2i}\mid \eta_{2i})\,
  \phi(u_i)\; \mathrm{d}u_i
\;\approx\;
\frac{1}{\sqrt{\pi}} \sum_{k=1}^{K} w_k\,
  p_1\!\big(y_{1i}\mid X_{1i}\beta_1 + \lambda_1 \sqrt{2}\,z_k\big)\,
  p_2\!\big(y_{2i}\mid X_{2i}\beta_2 + \lambda_2 \sqrt{2}\,z_k\big).
```

The quadrature makes the marginal smooth and `ForwardDiff`-friendly, so the fit
is a plain L-BFGS optimisation with forward-mode autodiff. For identifiability,
`λ₁` is constrained positive (`λ₁ = exp(·)`) to remove the `u → -u` sign flip.

!!! note "Gaussian × Gaussian ≡ the rho12 model"
    When **both** axes are Gaussian the marginal is exactly bivariate normal, so
    the marginal log-likelihood and the reported correlation reduce to the
    residual-correlation
    [`rho12`](tutorials/bivariate-coscale.md) model. In that case the two
    loadings are not separately identified (a flat `λ₁` ridge), but the marginal
    covariance — hence the log-likelihood and `ρ` — is. For Gaussian ×
    non-Gaussian (no free residual variance on the non-Gaussian axis), all
    parameters are identified.

## The latent-scale correlation

The loadings `λ₁`, `λ₂` are not directly comparable across families, because each
family contributes its own observation-level noise on its link scale. To put the
dependence on a single, interpretable scale we standardise it the way
Nakagawa & Schielzeth (2010) standardise variance components — adding each
family's **link-scale residual variance** `v_k` to the latent variance it
already carries:

```math
\rho = \frac{\lambda_1 \lambda_2}
            {\sqrt{(\lambda_1^2 + v_1)\,(\lambda_2^2 + v_2)}}.
```

The numerator `λ₁ λ₂` is the cross-axis covariance contributed by the shared
latent; each denominator term `λ_k² + v_k` is the total latent-scale variance of
axis `k`. `v_k` comes from [`DRModels.link_residual`](@ref):

- **Gaussian** (identity link): `v = σ²`, the fitted residual variance.
- **Poisson** (log link): `v = log(1 + 1/μ̄)`, evaluated at a representative
  fitted mean `μ̄`.
- **Binomial** (logit link): `v = π²/3` (distribution-free).
- **Beta** (logit link): `v = trigamma(μ̄ φ) + trigamma((1-μ̄) φ)`, `φ` the precision.
- **Gamma** (log link): `v = trigamma(1/φ)`.
- **NegBinomial2** (log link): `v = trigamma(θ)`, `θ` the size.

Because `ρ` is a ratio of the same latent variance on top and bottom, it is
rotation-invariant and lies in `(-1, 1)`. It is a **reporting** quantity: it is
computed from the fitted parameters and never enters the objective.

!!! note "Why `v = σ²` for the Gaussian axis here"
    In this shared-latent parameterisation the Gaussian residual lives in `v`
    directly (there is no separate `Ψ` as in GLLVM/gllvmTMB, which report `v = 0`
    for Gaussian because the residual sits elsewhere). Using `v = σ²` is what
    makes the Gaussian × Gaussian case collapse onto the `rho12` model; using
    `0` would force `ρ = 1`.

## Confidence intervals for ρ

The cross-family fit can return three intervals for `ρ`, each on the
correlation scale:

| Field             | Method                                   | When to use |
|-------------------|------------------------------------------|-------------|
| `rho_ci_wald`     | Fisher-z (delta method on `atanh ρ`)     | Always computed (`confint = true`); cheapest. |
| `rho_ci_profile`  | Profile likelihood (`profile = true`)    | **Recommended** — best calibrated near the boundary. |
| `rho_ci_boot`     | Parametric bootstrap (`B = …` refits)    | Most robust; most expensive. |

- **Fisher-z Wald** applies the delta method to `atanh(ρ(θ))` using the observed
  information (the Hessian of the marginal negative log-likelihood), then maps
  the symmetric `atanh`-scale interval back through `tanh`. The `atanh`
  transform keeps the interval inside `(-1, 1)`. Returns `(NaN, NaN)` if the
  Hessian is not invertible.

- **Profile likelihood** (recommended) re-optimises the model under a soft
  constraint that fixes `ρ(θ)` at a trial value on the `atanh` scale, and bisects
  for the values where the deviance rises by the `χ²(1)` quantile. It is better
  calibrated than Wald when `ρ` is near `±1`, and cheaper than the bootstrap.

- **Parametric bootstrap** resamples the shared latent and the per-family draws
  at the fitted parameters, refits each replicate, and takes the percentile
  interval of the resulting `ρ̂`. It makes the fewest distributional assumptions
  but costs `B` extra fits.

## Worked example: Gaussian × Poisson

We simulate a continuous response (Gaussian) and a count (Poisson) that share a
latent, fit the cross-family model through `drm(...)`, and read off the
latent-scale correlation with its profile-likelihood interval.

```@example xfam
using DRModels, Random, Statistics
Random.seed!(2024)

n = 800
u = randn(n)                       # shared per-observation latent, u ~ N(0,1)
x = randn(n)                       # one shared covariate, on both axes
X1 = hcat(ones(n), x)
X2 = hcat(ones(n), x)

# True parameters.
β1, β2 = [1.0, 0.4], [0.2, 0.3]    # fixed effects (Gaussian, Poisson)
λ1, λ2 = 0.8, 0.5                  # latent loadings

η1 = X1 * β1 .+ λ1 .* u
η2 = X2 * β2 .+ λ2 .* u
y1 = η1 .+ randn(n)                                            # Gaussian, σ = 1
y2 = [Float64(rand(DRModels.Distributions.Poisson(exp(η2[i])))) for i in 1:n]  # Poisson
dat = (; y1, y2, x)

form = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
          sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1))
fit = drm(form, (Gaussian(), Poisson()); data = dat, K = 32, profile = true)

fit.converged        # did L-BFGS converge?
```

The fitted loadings and the link-scale residual variances feed the latent-scale
correlation:

```@example xfam
(λ1 = round(fit.λ1, digits = 3), λ2 = round(fit.λ2, digits = 3),
 v1 = round(fit.v1, digits = 3), v2 = round(fit.v2, digits = 3))
```

`v1` is the fitted Gaussian residual variance (the true value is `σ² = 1`);
`v2 = log(1 + 1/μ̄)` is the Poisson axis' link-scale variance at its
representative mean. The latent-scale
correlation and its **profile-likelihood** interval are:

```@example xfam
round(fit.rho_latent, digits = 3)
```

```@example xfam
round.(fit.rho_ci_profile, digits = 3)        # recommended interval for ρ
```

The point estimate matches the value implied by the simulation truth,
`ρ = λ₁λ₂ / sqrt((λ₁²+v₁)(λ₂²+v₂)) ≈ 0.36`, and the profile interval covers it.
The cheaper Fisher-z Wald interval is also available:

```@example xfam
round.(fit.rho_ci_wald, digits = 3)           # Fisher-z delta-method interval
```

For the bootstrap interval, pass `B` (number of refits); it is omitted here to
keep the page fast to build:

```julia
fit_boot = drm(form, (Gaussian(), Poisson()); data = dat, K = 32, B = 500)
fit_boot.rho_ci_boot       # percentile interval from 500 parametric-bootstrap refits
```

## Returned fields

The cross-family `drm(...)` call returns a `NamedTuple`. Its
dependence-related fields are:

| Field             | Meaning |
|-------------------|---------|
| `rho_latent`      | Latent-scale correlation `ρ` (the headline estimate). |
| `rho_ci_wald`     | Fisher-z Wald interval `(lo, hi)` for `ρ` (or `(NaN, NaN)`). |
| `rho_ci_profile`  | Profile-likelihood interval (recommended); `(NaN, NaN)` unless `profile = true`. |
| `rho_ci_boot`     | Parametric-bootstrap interval; `(NaN, NaN)` unless `B > 0`. |

The fit also returns the fixed effects `β1`/`β2`, the loadings `λ1`/`λ2`, the
Gaussian residual SDs `σ1`/`σ2` (`NaN` on non-Gaussian axes), the link-scale
variances `v1`/`v2`, the `loglik`, `converged`, and `iterations`.

## See also

- [Changing residual coupling with rho12](tutorials/bivariate-coscale.md) — the
  Gaussian × Gaussian residual-correlation model that this generalises.
- [`DRModels.link_residual`](@ref) — the per-family link-scale variance `v_k`.

## API

```@docs
DRModels.link_residual
```

### Post-fit accessors

```@docs
mf_coef
mf_summary
mf_fitted
mf_aic
mf_bic
```

## References

- Nakagawa, S. & Schielzeth, H. (2010). Repeatability for Gaussian and
  non-Gaussian data: a practical guide for biologists.
  *Biological Reviews*, 85(4), 935–956.
