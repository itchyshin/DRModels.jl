# Laplace vs variational marginals

!!! note "Status — Experimental"
    A variational (VA / ELBO) marginal is an **opt-in Experimental** alternative
    for random-intercept `(1 | g)` models. **`:LA` remains the default.** For a
    scalar Poisson random intercept, `:LA` uses fixed, non-adaptive 32-node
    Gauss–Hermite quadrature, not one-point Laplace or adaptive GHQ. Other
    `:LA` routes are not identified by this statement.

    **Available scope:** Poisson, Binomial, NegBinomial2,
    Gamma, and Beta `(1 | g)` via `drm(...; marginal = :VA)` (scale families need
    `sigma ~ 1`). That `loglik` is an ELBO, not a Laplace log-likelihood. Mixed
    LA/VA AIC / LRT errors. Phylo / crossed / correlated slopes / ZI / hurdle /
    `sigma ~ x` are not supported. In the evaluated Gamma random-intercept
    model, LA and VA agree on shape while LA is faster.

## What "the marginal" is, and why it matters

When a model has random effects `z`, `drm` does not maximise the joint
likelihood of data and random effects directly. It integrates the random
effects out, leaving a *marginal* likelihood that depends only on the fixed
parameters and the variance components:

```
L(θ) = ∫ p(y | z, θ) p(z | θ) dz.
```

That integral has no closed form for non-Gaussian families, so it must be
approximated. The quality of the approximation is not a side detail: it is what
the dispersion, shape, and zero-inflation parameters are estimated *against*. A
biased marginal biases exactly those parameters.

### The default `:LA` route

For a scalar Poisson random intercept `(1 | g)`, the public default `:LA` route
uses fixed, non-adaptive 32-node Gauss–Hermite quadrature. It is neither
one-point Laplace nor adaptive Gauss–Hermite quadrature. A Laplace approximation
in general replaces the integrand with a Gaussian centred at the posterior mode of `z`,
matched in curvature (the Hessian) at that mode. It is exact only when the
responses are Gaussian, a Gaussian random effect enters the mean linearly, and
the residual variance is independent of that random effect; then the integrand
is Gaussian. A mean random intercept in that Gaussian model needs no
approximation.

The trouble starts when the integrand is **not** close to Gaussian:

- **Skewed or heavy-tailed posteriors** — a single mode-plus-curvature match
  understates the mass in the tail, so the integral, and the variance/shape
  parameters tied to it, are off.
- **Multimodal posteriors** — LA sees one mode and is blind to the others; which
  mode it lands on can depend on the optimiser, the OS, or the BLAS.
- **Dispersion / shape / zero-inflation parameters** — these read the *shape* of
  the integrand, not just its peak, so they absorb the approximation error
  first. Mean (location) parameters are comparatively robust.

## Why the approximation can matter

Related latent-variable models in GLLVModels.jl illustrate two geometries in
which a one-mode curvature approximation can be unreliable:

- **Two-part Gamma shape.** In a two-part (hurdle) Gamma model, the Gamma shape
  parameter `α` was recovered roughly **7× too low** under Laplace. The mean was
  fine; the shape — which is read off the curvature of a skewed positive density
  — was badly biased.
- **ZINB multimodality.** In a zero-inflated negative binomial model the
  zero-inflation probability `π` and the low-count-mean intercept `βc` trade off:
  a zero can be "structural" (`π`) or "a Poisson/NB zero from a small mean"
  (`βc`). That gives the marginal **two modes**, and the sign of the count
  intercept was observed to **flip across OS / BLAS** — a hallmark of LA picking
  different modes on different platforms.

Neither failure is a bug in the optimiser; both are the geometry the Laplace
approximation cannot see. They motivate checking the marginal approximation,
but they do not show that DRModels.jl's experimental VA route fixes those
specific models.

## The variational (VA / ELBO) alternative

The variational path replaces "find one mode and match curvature" with "fit a
whole approximating distribution." We choose a factorised Gaussian

```
q(z) = N(m, diag(v)),
```

and pick `m` and `v` to maximise the **evidence lower bound** (ELBO):

```
ELBO(θ, m, v) = E_q[ log p(y, z | θ) ] − E_q[ log q(z) ]  ≤  log L(θ).
```

The ELBO is a *provable lower bound* on the true log marginal for a fixed
variational distribution. Its optimisation can still have local optima, and a
factorised Gaussian `q` need not represent a multimodal posterior. The bound is
an objective property, not a guarantee of global optimisation or complete
posterior geometry.

The expectations under a Gaussian `q` are tractable in the two regimes DRModels.jl
needs:

- **Closed form** when the log-density is linear in the linear predictor `η` and
  in `e^{±η}` — this covers **Poisson** and **Gamma**, because the Gaussian
  expectations of `η` and of `e^{η}` (a log-normal moment) are both analytic.
- **One-dimensional Gauss–Hermite quadrature** for everything else —
  **Binomial**, **negative binomial**, and **Beta** — where the expectation
  reduces to a single integral over the scalar `η`, cheaply and accurately
  evaluated with a handful of GH nodes.

Because `q` carries a *variance* `v`, not just a location, it represents tail and
spread directly, so the shape and dispersion parameters are no longer estimated
against a curvature match at a single point.

## When to use which

| Situation | Recommendation |
|---|---|
| Fixed-effects-only model | VA adds nothing — there is no latent integral to approximate. |
| Gaussian response with a Gaussian RE entering the mean linearly and independent residual variance | VA adds nothing — the marginal is already exact here. |
| Ordinary Gamma `(1\|g)` shape | LA ≈ VA in the scoped Gamma comparison; **prefer LA** (15–20× faster warm). |
| Two-part / hurdle / ZINB geometry | VA may help in principle, but **DRModels.jl does not currently support these VA models**. |
| Speed-critical fits | Route-specific: `:LA` is the default; Poisson scalar random intercepts use fixed GHQ-32. |

In short: **`:LA` is the default and its numerical implementation is
route-specific.** On the public Gamma random-intercept cell, Julia matches the
R/TMB pattern: the two marginals agree on `α` and LA wins on time
in the evaluated Gamma comparison. The two-part and zero-inflated models where
VA could be most useful are not currently supported here.

## The public API (Experimental)

The marginal is selected with `marginal` (not Gaussian `method = :ML/:REML`).
`:LA` remains the default integration route; its numerical implementation is
route-specific:

```julia
# default `:LA` route; integration is route-specific
drm(...; marginal = :LA)

# opt-in variational marginal (Experimental: `(1 | g)` on Poisson / Binomial /
# NegBinomial2 / Gamma / Beta; scale families need sigma ~ 1)
drm(...; marginal = :VA)
```

Everything else about the call — the `bf(...)` formulas, the family, the data —
stays the same; only how the random effects are integrated out changes.
`method = :VA` on non-Gaussian families is rejected with a pointer to `marginal`.

## How to assess the approximation

The Experimental `(1 | g)` path is checked against three mathematical
expectations with known outcomes, rather than judged only by whether the fitted
numbers look plausible:

1. **Variance → 0 collapses to independence.** As the random-effect variance is
   driven to zero there is nothing left to integrate, so the ELBO equals the
   ordinary independent log-likelihood. This pins the no-RE limit exactly.
2. **ELBO ≤ dense quadrature.** At low latent dimension the true marginal is
   computed by dense *adaptive* Gauss–Hermite. The ELBO, being a lower bound,
   must sit at or below it — never above. (Non-adaptive engine GHQ centred at 0
   can sit below the ELBO; that is not a counterexample.)
3. **Family limits.** The negative binomial becomes Poisson as its size
   `r → ∞`, so NB2-VA should approach Poisson-VA when both are applied to the
   same simulated data.

On Gamma `(1 | g)`, the evaluated comparison finds LA ≈ VA on shape `α` and LA is
much faster. VA beyond random
intercepts (phylo / crossed / ZI / hurdle) and two-part models is not supported
here.

## How this differs from drmTMB

drmTMB is built on TMB, which is **Laplace-only**. Offering a variational
marginal alongside LA is therefore a Julia-only option; a VA result should not
be interpreted as reproducing drmTMB's marginal likelihood. The option is most
relevant where a one-mode approximation is scientifically questionable, such as
two-part shape or ZINB multimodality, although those VA models are not currently
supported in DRModels.jl. On ordinary Gamma `(1 | g)`, the evaluated comparison
does **not** show a VA accuracy edge; prefer the default Laplace route, as in R.

## See also

- [Which scale are you modelling?](which-scale.md) · [Improving convergence](convergence.md)
