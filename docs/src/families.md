# Response families

!!! note "Status — Stable"
    Every family below is fitted by the verified `drm` front end, and every
    worked snippet on this page runs. This is a *reference catalogue* with one
    minimal fit per family; for the modelling rationale and side-by-side R
    parity see [Choosing response families](model-guides/distribution-families.md)
    and the per-topic tutorials.

You pick a family from the **shape** of your response, then give each of its
parameters a formula with `bf`. The mean **μ** always comes from the response
formula (`y ~ …`); the family's second parameter lives in the `sigma` slot
(`sigma ~ …`) with a family-specific link, and some families add further
parameters (`nu`, `zoi`, `coi`, ordinal cutpoints).

| Response looks like… | Family | Mean link | `sigma` slot |
|---|---|---|---|
| Real-valued, symmetric | `Gaussian()` | identity | residual SD `σ` (log) |
| Real-valued, heavy tails | `Student()` | identity | scale `σ` (log) + d.o.f. `ν` (log) |
| Strictly positive, multiplicative | `LogNormal()` | identity on `log y` | SD of `log y` (log) |
| Strictly positive, right-skewed | `Gamma()` | log | shape via `α = 1/σ²` |
| Positive **with exact zeros** | `Tweedie()` | log | √dispersion `σ` + power `ν ∈ (1,2)` |
| Counts (variance ≈ mean) | `Poisson()` | log | — |
| Counts, overdispersed | `NegBinomial2()` | log | dispersion `θ` (log) |
| Positive counts (zero-truncated) | `TruncatedNegBinomial2()` | log | dispersion `θ` (log) |
| Positive counts, no overdispersion | `TruncatedPoisson()` | log | — |
| Proportions in `(0,1)` | `Beta()` | logit | precision via `φ = 1/σ²` |
| Successes out of trials | `Binomial()` | logit | — (`cbind(s,f)` or `0/1`) |
| Successes, overdispersed | `BetaBinomial()` | logit | precision via `φ = 1/σ²` |
| Proportions on `[0,1]` incl. 0/1 | `ZeroOneBeta()` | logit | precision `φ` + `zoi` / `coi` |
| Ordered categories `1..K` | `CumulativeLogit()` | logit cutpoints | — (K−1 cutpoints) |

A handful of families share the name of a `Distributions.jl` distribution
(`Gaussian` excepted: `Poisson`, `Binomial`, `Beta`, `Gamma`, `LogNormal`).
DRModels exports its own *family* object of that name; when a snippet also needs the
distribution to simulate, it qualifies it as `Distributions.Poisson` etc.

## Continuous, real-valued

### Gaussian — `Gaussian()`

Identity link on the mean, log link on the residual SD `σ`. The textbook
location–scale model: both the mean and the spread can depend on covariates.

```@example fam
using DRModels, Random
import Distributions          # qualified, for the simulating distributions
Random.seed!(1)

n = 600
x = randn(n)
y = (1.0 .+ 0.5 .* x) .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))
(mu = coef(fit, :mu), logsigma = coef(fit, :sigma))   # ≈ (1.0, 0.5) and (-0.3, 0.4)
```

### Student-t — `Student()`

The robust sibling of Gaussian: identity μ, log `σ`, plus degrees of freedom `ν`
(log link) that govern the tail weight. Small `ν` downweights outliers; `ν → ∞`
returns to Gaussian.

```@example fam
Random.seed!(2)
yt = 0.5 .+ 0.8 .* x .+ 2.0 .* rand(Distributions.TDist(4), n)
fitt = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = (; y = yt, x))
(slope = coef(fitt, :mu)[2], nu = 2 + exp(coef(fitt, :nu)[1]))   # d.o.f., ν = 2 + exp(η)
```

### LogNormal — `LogNormal()`

For strictly positive, multiplicative data. The μ formula is the mean of
`log y`; `σ` (log link) is the SD of `log y`. The response-scale median is
`exp(μ)`. A random intercept `(1 | g)` or correlated slope `(1 + x | g)`, or a
`phylo(1 | group)` (needs `tree = ...`) / `relmat(1 | group)` (needs
`K = ...`) structured marker, may be placed on the mean; the structured
routes delegate to `Gaussian()` on `log y` (exact — no approximation) and
shift the reported log-likelihood by the parameter-free Jacobian. `animal`/
`spatial` markers are not implemented for `LogNormal()`.

```@example fam
Random.seed!(3)
yln = exp.(0.5 .+ 0.3 .* x .+ 0.4 .* randn(n))
fitln = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = (; y = yln, x))
(intercept = coef(fitln, :mu)[1], sd_logy = exp(coef(fitln, :sigma)[1]))   # ≈ (0.5, 0.4)
```

### Gamma — `Gamma()`

Strictly positive, right-skewed (durations, sizes, concentrations). Log link on
the mean; the `sigma` slot maps to the shape `α = 1/σ²`, so the coefficient of
variation is `1/√α`. Recover the shape as `exp(-2·log σ)`.

```@example fam
Random.seed!(4)
αsh = 8.0
μg  = exp.(0.5 .+ 0.4 .* x)
yg  = Float64.([rand(Distributions.Gamma(αsh, μi / αsh)) for μi in μg])
fitg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma(); data = (; y = yg, x))
(mean_ratio = exp(coef(fitg, :mu)[2]), shape = exp(-2 * coef(fitg, :sigma)[1]))   # shape ≈ 8
```

### Tweedie — `Tweedie()`

Positive data **with exact zeros** (biomass, rainfall, total loss): a point mass
at 0 plus a positive continuous part. Log link on the mean; `sigma` is the
√dispersion (`φ = σ²`); `nu` is the power `p ∈ (1,2)` on a logit link
(`p = 1 + logistic(coef(:nu))`).

```@example fam
Random.seed!(5)
function rtw(μi, φ, p)                              # compound Poisson–Gamma draw
    λ = μi^(2 - p) / (φ * (2 - p)); γ = φ * (p - 1) * μi^(p - 1)
    N = rand(Distributions.Poisson(λ))
    N == 0 ? 0.0 : rand(Distributions.Gamma(N * (2 - p) / (p - 1), γ))
end
ytw = [rtw(exp(0.5 + 0.3 * xi), 2.0, 1.5) for xi in x]
fittw = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = (; y = ytw, x))
(zeros = count(==(0.0), ytw), power = 1 + 1 / (1 + exp(-coef(fittw, :nu)[1])))   # power ≈ 1.5
```

An ordinary `(1 | g)` random intercept, and an independent `(0 + x | g)`
random slope, on `mu` are both supported (32-node Gauss–Hermite marginal,
matching drmTMB's `tweedie() mu ~ x + (1 | g)` / `(0 + x | g)`); the group
SD is `exp(coef(fit, :resd)[1])` either way. The CORRELATED random slope
`(1 + x | g)`, random effects on `sigma`/`nu`, and structured (phylo/relmat)
markers on `mu` remain unimplemented (drmTMB rejects the same forms).

## Counts

### Poisson — `Poisson()`

Counts with variance ≈ mean. Log link on the mean; no scale parameter. Slopes
are log rate ratios (`exp(β)` is the multiplicative effect on the expected
count).

```@example fam
Random.seed!(6)
λp = exp.(0.3 .+ 0.5 .* x)
yp = Float64.([rand(Distributions.Poisson(λi)) for λi in λp])
fitp = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = yp, x))
(logmu = coef(fitp, :mu), rate_ratio = exp(coef(fitp, :mu)[2]))   # rate ratio ≈ 1.65
```

### Negative-binomial — `NegBinomial2()`

Overdispersed counts. Log link on the mean and a dispersion `θ` in the `sigma`
slot (log link); variance `= μ + μ²/θ`, so `θ → ∞` recovers Poisson.

```@example fam
Random.seed!(7)
θnb = 2.5
μnb = exp.(0.4 .+ 0.5 .* x)
ynb = Float64.([rand(Distributions.NegativeBinomial(θnb, θnb / (θnb + μi))) for μi in μnb])
fitnb = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y = ynb, x))
exp(-2 * coef(fitnb, :sigma)[1])      # estimated dispersion θ = 1/σ² ≈ 2.5
```

The `sigma` coefficient is on the log-`σ` scale, and the NB2 dispersion is
`θ = 1/σ² = exp(-2·coef(:sigma))` (not `exp(coef(:sigma))`).

`TruncatedNegBinomial2()` is the same family conditioned on `y ≥ 1`, for counts
that are positive by construction (litter sizes, group sizes given presence).
`TruncatedPoisson()` is the Poisson conditioned on `y ≥ 1`; adding an `hu ~ ...`
part to either truncated family (or to `Poisson()` / `NegBinomial2()`) fits the
hurdle model.

## Proportions and binary data

### Binomial — `Binomial()`

Successes out of known trials (logistic regression). Logit link on the success
probability; no dispersion parameter. The response is either a `cbind(successes,
failures)` two-column term or a plain `0/1` Bernoulli vector.

```@example fam
Random.seed!(8)
ntrials = fill(20, n)
pb = 1 ./ (1 .+ exp.(-(0.2 .+ 0.7 .* x)))
succ = Float64.([rand(Distributions.Binomial(ntrials[i], pb[i])) for i in 1:n])
fail = ntrials .- succ
fitb = drm(bf(@formula(cbind(succ, fail) ~ x)), Binomial(); data = (; succ, fail, x))
coef(fitb, :mu)      # logit-scale intercept & slope ≈ (0.2, 0.7)
```

### Beta — `Beta()`

Proportions strictly inside `(0,1)`. Logit link on the mean; the `sigma` slot
carries `σ` with precision `φ = 1/σ²` (recover precision as `exp(-2·log σ)`).

```@example fam
Random.seed!(9)
μbe = 1 ./ (1 .+ exp.(-(0.3 .+ 0.6 .* x))); φbe = 12.0
ybe = Float64.([rand(Distributions.Beta(μi * φbe, (1 - μi) * φbe)) for μi in μbe])
fitbe = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Beta(); data = (; y = ybe, x))
(slope = coef(fitbe, :mu)[2], precision = exp(-2 * coef(fitbe, :sigma)[1]))   # φ ≈ 12
```

### Beta-binomial — `BetaBinomial()`

Successes out of trials **with extra-binomial overdispersion**. Logit mean;
precision `φ = 1/σ²` in the `sigma` slot. Requires a `cbind(successes,
failures)` response.

```@example fam
Random.seed!(10)
ntr = fill(30, n)
μbb = 1 ./ (1 .+ exp.(-(0.1 .+ 0.5 .* x))); φbb = 5.0
sbb = Float64.([rand(Distributions.BetaBinomial(ntr[i], μi * φbb, (1 - μi) * φbb)) for (i, μi) in enumerate(μbb)])
fbb = ntr .- sbb
fitbb = drm(bf(@formula(cbind(s, f) ~ x), @formula(sigma ~ 1)), BetaBinomial(); data = (; s = sbb, f = fbb, x))
(slope = coef(fitbb, :mu)[2], precision = exp(-2 * coef(fitbb, :sigma)[1]))   # φ ≈ 5
```

### Zero-one-inflated beta — `ZeroOneBeta()`

Proportions on the **closed** interval `[0,1]` (values may be exactly 0 or 1).
Beta mean (logit) and precision `φ` on the interior, plus `zoi` (logit
probability the value is a boundary) and `coi` (logit probability of a 1 given a
boundary).

```@example fam
Random.seed!(11)
function rzob(xi)
    μi = 1 / (1 + exp(-(0.2 + 0.5 * xi))); φ = 10.0; zoi = 0.15; coi = 0.4
    rand() < zoi ? (rand() < coi ? 1.0 : 0.0) : rand(Distributions.Beta(μi * φ, (1 - μi) * φ))
end
yzob = [rzob(xi) for xi in x]
fitzob = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)), ZeroOneBeta(); data = (; y = yzob, x))
(boundaries = count(yi -> yi == 0 || yi == 1, yzob), zoi = 1 / (1 + exp(-coef(fitzob, :zoi)[1])))
```

## Ordered categories

### Cumulative-logit — `CumulativeLogit()`

Ordered categories coded `1, 2, …, K` (Likert ratings, severity classes):
`Pr(y ≤ k) = logistic(θ_k − η)` with ordered cutpoints. The single linear
predictor comes from the `mu` formula **with its intercept dropped** (the
cutpoints absorb the level), so `coef(fit, :mu)` is the slope.

```@example fam
Random.seed!(12)
θcut = [-1.0, 0.0, 1.2]; K = 4
yo = map(x) do xi
    u = rand(); η = 0.8 * xi; cat = K
    for j in 1:(K-1)
        (u < 1 / (1 + exp(-(θcut[j] - η)))) && (cat = j; break)
    end
    cat
end
fito = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y = Float64.(yo), x))
coef(fito, :mu)[1]      # slope ≈ 0.8
```

An ordinary random intercept `(1 | g)` or an independent random slope
`(0 + x | g)` on `mu` is integrated out by 32-node Gauss–Hermite quadrature
(the same scheme as the Poisson/Gamma/Tweedie GLMM routes); the correlated
form `(1 + x | g)` is not implemented.

```julia
fit_re = drm(bf(y ~ x + (1 | g)), CumulativeLogit(); data = dat)
exp(coef(fit_re, :resd)[1])   # random-intercept SD
```

An unlabelled, intercept-only phylogenetic random intercept `phylo(1 | species)`
on `mu` is fit via the sparse augmented-state Laplace GLMM route (the same
engine as the Poisson/Gamma/Binomial/Beta `phylo(1 | g)` cells), matching
drmTMB 0.7.0's `validate_ordinal_phylo_mu_structured_term()`. It cannot be
combined with an ordinary `(1 | g)`/`(0 + x | g)` random effect on `mu`.

```julia
fit_phylo = drm(bf(y ~ x + phylo(1 | species)), CumulativeLogit(); data = dat, tree = tr)
re_sd(fit_phylo)[:species]   # phylogenetic SD, raw branch-length scale
```

## See also

- [Getting started](getting-started.md) — install and a first Gaussian fit.
- [Choosing response families](model-guides/distribution-families.md) — the
  modelling rationale, with R parity.
- [What can I fit today?](model-guides/model-map.md) — the live capability map,
  including the `zi` / `hu` count modifiers and random-effect support per family.
