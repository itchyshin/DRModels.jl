# Count abundance and extra zeros

!!! note "Status — Stable (Poisson + NB2 + zi + hu)"
    Mirrors drmTMB's [Count abundance and extra zeros](https://itchyshin.github.io/drmTMB/articles/count-nbinom2.html).
    **In DRModels.jl today:** the **Poisson** family `Poisson()`, the
    **negative-binomial** family `NegBinomial2()` (overdispersed counts), and both
    the **`zi` zero-inflation** and **`hu` hurdle** modifiers on either count
    family (ZIP / ZINB and hurdle-Poisson / hurdle-NB).

Counts — abundances, visit tallies, event counts — are non-negative integers, so
a Gaussian model is the wrong shape. The **Poisson** family models the log of the
expected count.

## A Poisson abundance model

```@example count
using DRModels, Random
import Distributions          # `Poisson` below is DRModels's family; qualify the distribution
Random.seed!(20260615)

n = 3000
x = randn(n)
λ = exp.(0.3 .+ 0.5 .* x)                       # log-mean increases with x
y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
dat = (; y, x)

fit = drm(bf(@formula(y ~ x)), Poisson(); data = dat)
coef(fit, :mu)              # coefficients on log λ
```

The slope is a **log rate ratio**: a one-unit increase in `x` multiplies the
expected count by `exp(β)`.

```@example count
exp(coef(fit, :mu)[2])      # rate ratio per unit x (≈ exp(0.5) ≈ 1.65)
```

Fitted values are returned on the response (count) scale:

```@example count
extrema(fitted(fit))        # fitted means λ̂ = exp(Xβ̂)
```

!!! tip "When Poisson is too tight"
    Poisson forces variance = mean. Real counts are often **overdispersed**
    (variance > mean). The negative-binomial family `NegBinomial2()` relaxes that
    — read on. (For **extra zeros**, see the `zi` and `hu` modifiers below.)

## Overdispersion: the negative-binomial (NB2) family

`NegBinomial2()` adds a dispersion (size) parameter `θ` carried on the log-`σ`
scale in the `sigma` slot: `θ = 1/σ² = exp(-2·coef(:sigma))`. The variance is
`μ + μ²/θ`, so smaller `θ` means heavier overdispersion and `θ → ∞` returns to
Poisson.

```@example count
Random.seed!(20260616)
θ = 2.5
μnb = exp.(0.4 .+ 0.5 .* x)
ynb = Float64.([rand(Distributions.NegativeBinomial(θ, θ / (θ + μi))) for μi in μnb])
datnb = (; y = ynb, x)

fitnb = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = datnb)
exp(-2 * coef(fitnb, :sigma)[1])     # estimated dispersion θ = 1/σ² (≈ 2.5 ⇒ real overdispersion)
```

A finite, smallish `θ` is the signal that the counts are overdispersed and a
plain Poisson would understate the uncertainty.

## Extra zeros: the `zi` modifier

Many count datasets have **more zeros** than any single count distribution
predicts (a site is unsuitable, a detector was off). Add a `zi ~ …` formula and
the model becomes a mixture — with probability `π` (logit link) the count is a
structural zero, otherwise it is drawn from the count family. Works on both
`Poisson()` (ZIP) and `NegBinomial2()` (ZINB):

```@example count
Random.seed!(20260619)
π = 0.40                                          # 40% structural zeros
λzi = exp.(0.6 .+ 0.4 .* x)
yzi = Float64.([rand() < π ? 0 : rand(Distributions.Poisson(λi)) for λi in λzi])
datzi = (; y = yzi, x)

fitzi = drm(bf(@formula(y ~ x), @formula(zi ~ 1)), Poisson(); data = datzi)
1 / (1 + exp(-coef(fitzi, :zi)[1]))     # recovered zero-inflation probability (≈ 0.40)
```

`coef(fitzi, :mu)` is still the log-mean of the *count* process; the `:zi` block
is the structural-zero model on the logit scale.

## A different two-part story: the `hu` hurdle modifier

A **hurdle** model splits the data in two: a logit "hurdle" decides zero vs
positive (`π = P(y = 0)`), and the positive counts follow the **zero-truncated**
count distribution. Unlike `zi`, *all* zeros come from the hurdle — use it when
"is it present at all?" and "how many, given present?" are distinct processes. Add
a `hu ~ …` formula (works on `Poisson()` and `NegBinomial2()`):

```@example count
Random.seed!(20260620)
πh = 0.40                                         # 40% structural (hurdle) zeros
λh = exp.(0.6 .+ 0.4 .* x)
rtpois(λ) = (while true; k = rand(Distributions.Poisson(λ)); k > 0 && return k; end)
yh = Float64.([rand() < πh ? 0 : rtpois(λi) for λi in λh])
dath = (; y = yh, x)

fith = drm(bf(@formula(y ~ x), @formula(hu ~ 1)), Poisson(); data = dath)
1 / (1 + exp(-coef(fith, :hu)[1]))      # recovered hurdle (zero) probability (≈ 0.40)
```

`zi` and `hu` cannot both be set on one model — they are competing stories for the
zeros.

## Counts that can't be zero: truncated NB2

Some counts are positive by construction — litter sizes, group sizes *given
presence*. `TruncatedNegBinomial2()` is the NB2 conditioned on `y ≥ 1`
(`P(k) = NB(k)/(1 − NB(0))`), with the same parameters as `NegBinomial2()`:

```@example count
Random.seed!(20260622)
rtnb(r, p) = (while true; k = rand(Distributions.NegativeBinomial(r, p)); k > 0 && return k; end)
μt = exp.(0.8 .+ 0.3 .* x); θt = 3.0
yt = Float64.([rtnb(θt, θt / (θt + μi)) for μi in μt])

fitt = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), TruncatedNegBinomial2(); data = (; y = yt, x))
exp(-2 * coef(fitt, :sigma)[1])      # recovered dispersion θ = 1/σ²
```

## Random effects: a count GLMM

Counts collected in groups (sites, individuals, broods) want a **random
intercept**. Add `(1 | g)` to the mean — the group effect on `log λ` is
integrated out per group by Gauss–Hermite quadrature, so `Poisson()` becomes a
proper count GLMM:

```@example count
Random.seed!(20260627)
Gc = 40; mc = 30; nc = Gc * mc
gc = repeat(1:Gc, inner = mc); xc = randn(nc)
bg = 0.6 .* randn(Gc)                                   # group random intercepts, SD 0.6
yc2 = Float64.([rand(Distributions.Poisson(exp(0.3 + 0.5 * xc[i] + bg[gc[i]]))) for i in 1:nc])

fitre = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = (; y = yc2, x = xc, g = gc))
re_sd(fitre)[:g]      # recovered group random-intercept SD (≈ 0.6)
```

`re_sd(fit)[:g]` is the group SD; `coef(fitre, :mu)` are the population-level
(`b = 0`) log-rate coefficients.

## See also

- [What can I fit today?](../model-guides/model-map.md) — the family/feature matrix.
- [Choosing response families](../model-guides/distribution-families.md).
