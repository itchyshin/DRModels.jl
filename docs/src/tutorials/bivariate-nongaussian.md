# Bivariate non-Gaussian models: joint or staged

!!! note "Status — mixed tiers"
    Mirrors drmTMB's [Bivariate non-Gaussian models](https://itchyshin.github.io/drmTMB/articles/bivariate-nongaussian.html).
    **In DRModels.jl today:** direct joint fits for two positive responses
    (`LogNormal()`) and two heavy-tailed responses (`Student()`), both Stable;
    and the staged frozen-margin route ([`associate_pairs`](@ref)), which is
    **Experimental** — see [API stability](../api-stability.md).

With two Gaussian responses, the residual correlation `ρ12` answers the
question, and
[Changing residual coupling with rho12](bivariate-coscale.md) covers it. With
two **non-Gaussian** responses the first decision is not computational but
scientific: *what kind of association do you want to estimate?*

Two routes, two different estimands:

| Your pair | Route | What the association means |
|---|---|---|
| Two positive traits, plausibly lognormal | `LogNormal()` joint fit | `rho12` = correlation of the two **log-response** residuals |
| Two heavy-tailed real-valued traits | `Student()` joint fit | `rho12` = **scatter** correlation of a shared-`ν` bivariate t |
| A mixed or discrete pair (e.g. continuous × binary) | [`associate_pairs`](@ref), staged | `eta` = latent-normal correlation **after freezing both margins** |

Neither `rho12` nor `eta` is the Pearson correlation of the two raw response
columns. That scale is part of the interpretation, not a detail — this page
shows the difference numerically.

## Two positive traits: the lognormal joint model

Both responses must be strictly positive. `log(y1)` and `log(y2)` are modelled
as bivariate normal, so `mu1`/`mu2` are means **on the log scale** and `sigma1`/
`sigma2` are log-scale SDs:

```math
(\log Y_{1i},\, \log Y_{2i}) \sim
N_2\!\left((\mu_{1i}, \mu_{2i}),\;
\begin{bmatrix}\sigma_1^2 & \rho_{12}\sigma_1\sigma_2\\
\rho_{12}\sigma_1\sigma_2 & \sigma_2^2\end{bmatrix}\right).
```

```@example bivng
using DRModels, Random, Statistics
Random.seed!(20260828)

n = 4000
x = randn(n)
ρ = 0.6                                    # true LOG-residual correlation
z1 = randn(n)
z2 = ρ .* z1 .+ sqrt(1 - ρ^2) .* randn(n)
y1 = exp.(0.3 .+ 0.5 .* x .+ 0.4 .* z1)    # strictly positive
y2 = exp.(-0.2 .+ 0.3 .* x .+ 0.5 .* z2)
dat = (; y1, y2, x)

fit_ln = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                rho12 = @formula(rho12 ~ 1)), LogNormal(); data = dat)

tanh(coef(fit_ln, :rho12)[1])      # log-residual ρ12  (≈ 0.6)
```

The log-response location and scale read back the same way as any Gaussian fit
(`mu1` on the identity link, `sigma1` through `exp`):

```@example bivng
(mu1 = coef(fit_ln, :mu1), sigma1 = exp(coef(fit_ln, :sigma1)[1]))
```

!!! warning "ρ12 is not the raw-scale correlation"
    On this simulation the fitted log-residual `ρ12` is ≈ 0.60 — the value that
    was simulated — while the Pearson correlation of the raw columns is a
    different number entirely:

    ```@example bivng
    (rho12_log = tanh(coef(fit_ln, :rho12)[1]), pearson_raw = cor(y1, y2))
    ```

    Both are legitimate summaries of the same data; they answer different
    questions. Quote the one you modelled.

Because the Jacobian of the log transform carries no parameters, this family
delegates the entire fit to the verified Gaussian engine on `log(y)`. That is
also why **structured markers work here**: `phylo(1 | g)`, `relmat(1 | g)`,
`animal(1 | g)`, and `spatial(1 | site)` are all supported, through exactly the
same routes as [the Gaussian bivariate model](bivariate-coscale.md).

## Two heavy-tailed traits: the Student-t joint model

The bivariate `Student()` model is a genuine multivariate t: one shared
scale-mixture drives both margins, so there is a single `ν`.

```@example bivng
using LinearAlgebra
using Distributions: Chisq
Random.seed!(7)

m = 4000
xs = randn(m)
R = [1.0 0.5; 0.5 1.0]                     # true SCATTER correlation
L = cholesky(R).L
w = rand(Chisq(6.0), m) ./ 6.0             # shared mixing, true ν = 6
Z = [L * randn(2) for _ in 1:m]
t1 = [ 1.0 + 0.4 * xs[i] + 0.8 * Z[i][1] / sqrt(w[i]) for i in 1:m]
t2 = [-0.5 + 0.2 * xs[i] + 1.0 * Z[i][2] / sqrt(w[i]) for i in 1:m]
dat_t = (y1 = t1, y2 = t2, x = xs)

fit_t = drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
               rho12 = @formula(rho12 ~ 1), nu = @formula(nu ~ 1)),
            Student(); data = dat_t)

(rho12 = tanh(coef(fit_t, :rho12)[1]),     # scatter correlation (≈ 0.5)
 nu    = 2 + exp(coef(fit_t, :nu)[1]))     # shared d.o.f., ν = 2 + exp(η)  (≈ 6)
```

Three conventions matter when reporting this fit, and each is easy to get wrong:

!!! warning "σ is a scale, not an SD"
    `sigma1`/`sigma2` are **scale** parameters of the t, not marginal standard
    deviations. The marginal SD is `σ·sqrt(ν/(ν−2))`:

    ```@example bivng
    ν̂ = 2 + exp(coef(fit_t, :nu)[1])
    σ̂1 = exp(coef(fit_t, :sigma1)[1])
    (scale1 = σ̂1, marginal_sd1 = σ̂1 * sqrt(ν̂ / (ν̂ - 2)))
    ```

    Comparing a t `σ` directly against a Gaussian `σ` compares two different
    quantities.

!!! note "ν uses the logm2 link"
    `ν = 2 + exp(η)`, which keeps `ν > 2` so the variance stays finite. Read it
    back with `2 + exp(...)`; a plain `exp(...)` under-reports `ν` by exactly 2
    and makes the tails look heavier than they are.

!!! warning "ρ12 = 0 is not independence"
    For any finite `ν` the two responses stay dependent even at `rho12 = 0`,
    because they share one mixing variable — extreme draws arrive together in
    both margins. Independence appears only in the Gaussian limit `ν → ∞`. A
    confidence interval for `rho12` that covers zero therefore does **not**
    license a claim of independence.

Structured markers (`phylo`/`relmat`/`animal`/`spatial`) are **deliberately not
available** for bivariate `Student()`. A Gaussian group-level effect under a
heavy-tailed conditional density has no closed-form marginal, and there is no
verified engine here whose per-leaf likelihood is bivariate-t; drmTMB defers the
same feature in its own `biv_student()`, so there is no reference implementation
on either side to mirror. This is a documented boundary, not an oversight —
fixed-effect fits are unaffected and parity-verified. For structured effects on
two heavy-tailed traits, model `log(y)` with `LogNormal()` if the responses are
positive, or use the Gaussian route and report the robustness caveat.

## A mixed pair: the staged association

When the two responses come from different families — a continuous trait and a
binary outcome, say — there is no shared residual covariance to write down.
[`associate_pairs`](@ref) fits the two margins **separately**, freezes them, and
then estimates a single latent-normal association `eta` conditional on those
frozen margins.

```@example bivng
Random.seed!(3)

k = 3000
xa = randn(k)
u1 = randn(k)
u2 = 0.6 .* u1 .+ sqrt(1 - 0.6^2) .* randn(k)   # true latent correlation 0.6
ya = 0.5 .+ 0.4 .* xa .+ u1                      # Gaussian margin
yb = Int.((-0.2 .+ 0.3 .* xa .+ u2) .> 0)        # Bernoulli margin
dat_a = (y1 = ya, y2 = yb, x = xa)

m1 = drm(bf(@formula(y1 ~ x), @formula(sigma ~ 1)), Gaussian(); data = dat_a)
m2 = drm(bf(@formula(y2 ~ x)), Binomial(); data = dat_a)

assoc = associate_pairs(m1, m2; kernel = latent_normal())
(pair_class = assoc.pair_class, eta = assoc.eta)     # eta ≈ 0.6
```

The kernel must be passed explicitly — there is no implicit association model,
exactly as in drmTMB.

!!! warning "The uncertainty is conditional on the frozen margins"
    This is a **two-stage estimator, not a joint model**. The margins are treated
    as exact, so the association's standard error ignores margin estimation
    error and is conditional. [`association`](@ref) reports that boundary in the
    result itself:

    ```@example bivng
    association(assoc).uncertainty
    ```

    DRModels.jl offers no simultaneous bands or profile intervals for `eta`, matching
    drmTMB, which bounds its own staged route the same way. Treat these
    intervals as experimental.

## Choosing between them

The routes are not interchangeable, and the choice is prior to the fitting:

- A **joint** model (`LogNormal()`, `Student()`) estimates both responses and
  their association *together*, so the association is a parameter of one
  likelihood and the margins inform it.
- The **staged** route estimates an association *given* margins that were fitted
  without it. That is a different estimand, and its uncertainty is conditional.

Prefer a joint fit when one exists for your pair. Reach for the staged route when
the two responses genuinely come from different families, and report the
conditional-uncertainty caveat alongside the estimate.

## See also

- [Changing residual coupling with rho12](bivariate-coscale.md) — the Gaussian
  bivariate model, including a predictor-dependent `ρ12` and the q=4
  phylogenetic coevolution engine.
- [Robust continuous responses](robust-student.md) — the single-response
  Student-t model, and the same `ν = 2 + exp(η)` convention.
- [Cross-family bivariate dependence](../cross-family.md) — two responses from
  *different* families through a shared latent (`DRModels.fit_mixed_family`).
- [API stability](../api-stability.md) — which of these surfaces are Stable and
  which are Experimental.
