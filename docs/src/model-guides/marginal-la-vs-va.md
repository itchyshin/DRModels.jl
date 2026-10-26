# Laplace vs variational marginals

!!! note "Status — Experimental"
    A variational (VA / ELBO) marginal is an **opt-in Experimental** alternative
    for random-intercept `(1 | g)` models. **`:LA` remains the default.** For
    one ordinary random intercept on the mean of a non-Gaussian family,
    `:LA` uses per-group *adaptive* Gauss–Hermite quadrature (5 nodes per group;
    21 for Tweedie, 41 for CumulativeLogit), not one-point Laplace. For a
    Gaussian random intercept on `sigma` it uses fixed, non-adaptive 32-node
    quadrature.

    **Available scope:** Poisson, Binomial, NegBinomial2,
    Gamma, and Beta `(1 | g)` via `drm(...; marginal = :VA)` (scale families need
    `sigma ~ 1`). That `loglik` is an ELBO, not a Laplace log-likelihood. Mixed
    LA/VA AIC / LRT errors. Phylo / crossed / correlated slopes / ZI / hurdle /
    `sigma ~ x` are not supported. In a retained three-seed smoke on one public
    Gamma random-intercept example, LA and VA gave similar shape estimates;
    warm VA fits took 15–21 times as long in the two timed replicates. This
    smoke does not establish general accuracy or coverage. See the
    [source-pinned report](https://github.com/itchyshin/DRModels.jl/blob/64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716/report/va-vs-laplace-bias.md).

## What "the marginal" is, and why it matters

When a model has random effects `z`, `drm` does not maximise the joint
likelihood of data and random effects directly. It integrates the random
effects out, leaving a *marginal* likelihood that depends only on the fixed
parameters and the variance components:

```
L(θ) = ∫ p(y | z, θ) p(z | θ) dz.
```

That integral has no closed form for non-Gaussian families, so it must be
approximated. The quality of the approximation matters because the dispersion,
shape, and zero-inflation parameters are estimated against that marginal. An
approximation error can affect those parameter estimates.

### The default `:LA` route

For one ordinary random intercept `(1 | g)` on the mean of a non-Gaussian
family (Poisson, Binomial, NegBinomial2, Gamma, Beta, BetaBinomial, Student-t),
the public default `:LA` route uses per-group **adaptive** Gauss–Hermite
quadrature with 5 nodes: each group's nodes are centred and scaled at that
group's posterior mode. Tweedie uses the same scheme with 21 nodes and
CumulativeLogit with 41. A Gaussian random intercept on `sigma`,
`sigma ~ 1 + (1 | g)`, instead uses fixed, non-adaptive 32-node quadrature
placed on the prior scale. A Laplace approximation replaces the integrand with
a Gaussian centred at the posterior mode of `z`, matched in curvature (the
Hessian) at that mode. For a Gaussian response with a Gaussian random effect
entering the mean linearly and residual variance independent of that effect,
the integrand is Gaussian and Laplace is guaranteed to be exact. A mean random
intercept in that model needs no approximation. In other models, Laplace is an
approximation and can incur error; accidental equality of a particular
integral is not ruled out.

### Matching drmTMB: `marginal = :Laplace`

drmTMB (TMB) integrates an ordinary random intercept by the one-point Laplace
approximation, so on `(1 | g)` the default `:LA` route (adaptive quadrature, 5
nodes per group) and drmTMB fit the same model with different integrators. On
seven Poisson, Beta, NegBinomial2, Gamma and Student test datasets their
log-likelihoods differed by 0.01 to 0.24 units, with drmTMB's always the one
further from a 61-node quadrature reference (a later Poisson comparison showed
a gap of about 0.0014); with `marginal = :Laplace` their log-likelihoods agree
to within about 1e-8 and their estimates to about 5e-8 (relative). In these
comparisons, the adaptive default was closer to the 61-node reference. This
does not establish a general accuracy ranking. Use `:Laplace` when matching
drmTMB's integration rule.

Before version 0.7.2, the ordinary mean-random-intercept `:LA` route used a
fixed, prior-centred 32-node grid; the current route uses adaptive, mode-centred
quadrature. Historical low-`sigma` comparisons for the old route are not
evidence about the current route. The retained smoke report below evaluates a
different, single public Gamma example and does not establish general accuracy
or coverage. Use `marginal = :Laplace` when the goal is to match drmTMB's
integration rule.

To reproduce drmTMB's numbers, request the Laplace approximation explicitly:

```julia
using DRModels
fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
          data = dat, marginal = :Laplace)
fit.marginal   # :Laplace
```

On these five families (Poisson, Binomial, NegBinomial2, Gamma and Beta),
`:Laplace` covers one ordinary `(1 | g)` on the mean (`sigma ~ 1` for the scale
families), by maximum likelihood. `Student()` accepts it too for one ordinary
`(1 | g)` on the mean (`sigma` and `nu` may carry fixed-effect formulas). A
Gaussian random intercept on `sigma`, `sigma ~ 1 + (1 | g)`, with a fixed-effect
mean also accepts it. `LogNormal()` accepts the selector for one ordinary mean
random intercept: this uses the same single mode-centred Gauss–Hermite node
as its default route and records the requested `:Laplace` label. The integral
is exact because `log(y)` is Gaussian given the random intercept. Other models with
`marginal = :Laplace` are refused.

!!! warning "A `sigma` formula with covariates breaks log-likelihood comparisons with drmTMB"
    For NegBinomial2, Gamma and Beta models with a random intercept and
    covariates in `sigma`, such as `sigma ~ mined` with `(1 | site)` on the
    mean, `marginal = :Laplace` is refused. The default `:LA` route then
    integrates with adaptive quadrature while drmTMB uses its Laplace
    approximation, so the two log-likelihoods differ by the integrator and are
    **not comparable**: on three real data sets (glmmTMB's Salamanders,
    grouseticks and Owls) the gap was 0.04 to 1.25 units. Do not use the
    difference for a likelihood-ratio test, AIC comparison or parity check
    across the two packages; compare the parameter estimates instead, or
    refit with `sigma ~ 1` and `marginal = :Laplace`, which reproduces drmTMB.
    A Laplace route for `sigma` covariates is tracked in [DRModels.jl#943](https://github.com/itchyshin/DRModels.jl/issues/943).

To test the random intercept, compare the `:Laplace` fit with the
fixed-effects model fitted as usual (no `marginal`): `lrtest(fixed, fit)`. The
fixed-effects log-likelihood is exact, so the pair is comparable. `lrtest`
refuses a `:LA` random-effect fit against a `:Laplace` one, because their
log-likelihoods differ by integration error as well as by model fit; refit both
with the same `marginal`.

The trouble starts when the integrand is **not** close to Gaussian:

- **Skewed or heavy-tailed posteriors** — a single mode-plus-curvature match can
  miss tail mass. The resulting integral and parameters tied to it may differ
  from their values under a more accurate integration rule.
- **Multimodal posteriors** — a one-mode curvature approximation does not
  represent other modes or their contribution to the integral.
- **Dispersion / shape / zero-inflation parameters** — these depend on more than
  the integrand's peak and can be sensitive to integration error. Mean
  (location) parameters can also be affected; the impact depends on the model
  and data.

## Why the approximation can matter

Related latent-variable models in GLLVM.jl illustrate two geometries in which
a one-mode curvature approximation may be unreliable:

- **Two-part Gamma shape.** A Gamma shape parameter can be sensitive to how a
  single-mode approximation represents a skewed positive density. The available
  DRModels.jl VA route does not fit two-part Gamma models, so this page does not
  claim that VA corrects their shape estimates.
- **ZINB multimodality.** In some zero-inflated negative-binomial data, the
  zero-inflation probability `π` and a low count mean can trade off: observed
  zeros can arise from either component, and the marginal likelihood may have
  multiple modes. This does not establish that every such model is multimodal
  or that DRModels.jl VA fixes the issue; these VA models are not supported here.

These are reasons to check the marginal approximation, not evidence that
DRModels.jl's experimental VA route fixes those specific models.

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
  reduces to a single integral over the scalar `η`, approximated with the
  configured Gauss–Hermite nodes.

Because `q` carries a *variance* `v`, it represents latent spread as well as
location. The factorised Gaussian approximation still cannot represent every
tail shape, dependence structure, or multiple modes, so the ELBO remains an
approximation to the marginal likelihood.

## When to use which

| Situation | Recommendation |
|---|---|
| Fixed-effects-only model | VA adds nothing — there is no latent integral to approximate. |
| Gaussian response with a Gaussian RE entering the mean linearly and independent residual variance | VA adds nothing — the marginal is already exact here. |
| Ordinary Gamma `(1\|g)` shape | LA and VA gave similar shape estimates in one three-seed smoke; warm VA fits were 15–21× slower in the two timed replicates. This is not a general accuracy, coverage, or runtime result; see the [source-pinned report](https://github.com/itchyshin/DRModels.jl/blob/64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716/report/va-vs-laplace-bias.md). |
| Two-part / hurdle / ZINB geometry | VA may help in principle, but **DRModels.jl does not currently support these VA models**. |
| Speed-critical fits | Route-specific: `:LA` is the default; non-Gaussian scalar random intercepts use per-group adaptive quadrature (5 nodes for most families). |

In short: **`:LA` is the default and its numerical implementation is
route-specific.** In the retained three-seed smoke on one public Gamma
random-intercept example, LA and VA gave similar `α` estimates and LA was
faster in the two warm replicates. That small comparison does not establish a
general accuracy or runtime ranking. The two-part and zero-inflated models
where VA could be useful are not currently supported here.

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
   must sit at or below it — never above. (A non-adaptive grid centred at 0, such
   as the engine's default before version 0.7.2, can sit below the ELBO; that is
   not a counterexample.)
3. **Family limits.** The negative binomial becomes Poisson as its size
   `r → ∞`, so NB2-VA should approach Poisson-VA when both are applied to the
   same simulated data.

Separately from these checks, every `:VA` fit verifies its own inner solves. Each
group's variational mean and variance come from an inner Newton solve, and at the
fitted parameters every group's solve must be stationary: its squared Newton
decrement, twice the ELBO gain still on offer, must be at most `1e-14`, or below
the rounding error of the group ELBO when the solve cannot go further. If one is
not, the reported ELBO is below the profiled ELBO and the optimiser did not follow
its gradient, so the fit is marked `converged = false` with a warning.
For example, in this small Poisson fit one group is all zeros:

```julia
y = [0, 0, 0, 0, 12, 147, 1778.0]; g = [1, 1, 1, 1, 2, 3, 4]
fit = drm(bf(@formula(y ~ 1 + (1 | g))), Poisson(); data = (; y, g), marginal = :VA)
fit.converged   # true, with every group's inner solve stationary
coef(fit)       # ≈ [2.837, 1.430]; ELBO ≈ -24.88
```

On one public Gamma `(1 | g)` example, the retained three-seed smoke found
similar LA and VA estimates of shape `α`, with LA faster in the two warm
replicates. It does not establish general accuracy, coverage, or runtime.
VA beyond random intercepts (phylo / crossed / ZI / hurdle) and two-part models
is not supported here. See the [source-pinned smoke report](https://github.com/itchyshin/DRModels.jl/blob/64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716/report/va-vs-laplace-bias.md).

## How this differs from drmTMB

drmTMB is built on TMB, which is **Laplace-only**. Offering a variational
marginal alongside LA is therefore a Julia-only option; a VA result should not
be interpreted as reproducing drmTMB's marginal likelihood. The option is most
relevant where a one-mode approximation is scientifically questionable, such as
two-part shape or ZINB multimodality, although those VA models are not currently
supported in DRModels.jl. On one ordinary Gamma `(1 | g)` example, the retained
three-seed smoke did **not** show a VA accuracy edge. That small comparison does
not rank accuracy or runtime generally. Use the default `:LA` for the current
adaptive route, or `marginal = :Laplace` to fit drmTMB's Laplace approximation.
See the [source-pinned report](https://github.com/itchyshin/DRModels.jl/blob/64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716/report/va-vs-laplace-bias.md).

## See also

- [Which scale are you modelling?](which-scale.md) · [Improving convergence](convergence.md)
