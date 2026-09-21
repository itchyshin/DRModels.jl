# Meta-analysis: known sampling variance, and the two correlations

!!! note "Status — Stable (univariate `meta_V`), Stable (bivariate `V =`)"
    Univariate meta-analysis via `meta_V(v)` has been in DRModels.jl since the
    Gaussian core. The **bivariate** known-sampling-covariance path accepts one
    known 2×2 block per study through `meta_vcov_bivariate` and the `V =`
    keyword. On matched inputs, its likelihood has been checked against
    drmTMB 0.7.0.

## 1. What makes meta-analysis different

In an ordinary regression every observation contributes the same unknown
residual variance, and the model estimates it. In meta-analysis each row is a
*study*, and each study arrives with its **own sampling variance already known**
— computed from that study's sample size and design, not estimated from the data
in front of you.

So the total variance of study ``i`` splits in two:

```math
\operatorname{Var}(y_i) = \underbrace{v_i}_{\text{known, sampling}} \; + \; \underbrace{\sigma^2}_{\text{estimated, heterogeneity}}
```

The quantity you want is almost always ``\sigma`` — the **between-study
heterogeneity**, how much the true effects genuinely differ once you have
accounted for the fact that small studies are noisy. Getting this split wrong in
either direction is the classic meta-analysis error: attribute sampling noise to
heterogeneity and you overstate genuine variation; absorb heterogeneity into the
sampling term and you understate it.

### Univariate

```julia
fit = drm(bf(@formula(yi ~ x + meta_V(vi)), @formula(sigma ~ 1)),
          Gaussian(); data = dat)
sigma(fit)      # the HETEROGENEITY SD -- not sqrt(v + sigma^2)
```

`meta_V(vi)` marks the data column holding the known sampling variances. The
fitted `sigma` is heterogeneity **alone**; the known ``v_i`` is added per row
inside the likelihood and never absorbed into it.

## 2. Two responses, and now *two* correlations

The moment you have two outcomes per study — say a mean difference in growth and
a mean difference in survival, measured on the same animals — a second
correlation appears, and the two are routinely conflated.

| | symbol | known or fitted? | what it is |
|---|---|---|---|
| **Sampling** correlation | ``r_i`` (inside ``V_i``) | **known** | the two effects were measured on the *same animals*, so their sampling errors covary |
| **Residual / heterogeneity** correlation | ``\rho_{12}`` | **fitted** | do studies with a larger true growth effect also have a larger true survival effect? |

Only the second is a scientific finding. The first is an artefact of the study
design — real, but not news. Per study:

```math
\begin{pmatrix} y_{1i} \\ y_{2i} \end{pmatrix}
\sim N\!\left(
\begin{pmatrix} \mu_{1i} \\ \mu_{2i} \end{pmatrix},\;
\underbrace{\begin{pmatrix} v_{1i} & c_i \\ c_i & v_{2i}\end{pmatrix}}_{V_i \text{ (known)}}
+
\underbrace{\begin{pmatrix} \sigma_1^2 & \rho_{12}\sigma_1\sigma_2 \\ \rho_{12}\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}}_{\text{heterogeneity (fitted)}}
\right)
```

```julia
V = meta_vcov_bivariate(v1, v2; cor12 = 0.6)   # known sampling correlation

fit = drm(bf(mu1    = @formula(y1 ~ x),
             mu2    = @formula(y2 ~ x),
             sigma1 = @formula(sigma1 ~ 1),
             sigma2 = @formula(sigma2 ~ 1),
             rho12  = @formula(rho12 ~ 1)),
          Gaussian(); data = dat, V = V)

fit.scales[:rho12][1]     # the HETEROGENEITY correlation
```

**If you omit `V`, `rho12` absorbs the sampling correlation** and you will report
a between-study correlation that is partly a measurement artefact. That failure
is silent — the fit converges and the number looks plausible. It is worth
constructing `V` even when you believe ``c_i`` is small.

### Supplying the sampling covariance

```julia
meta_vcov_bivariate(v1, v2)                      # independent sampling errors
meta_vcov_bivariate(v1, v2; cor12 = 0.6)         # a shared correlation
meta_vcov_bivariate(v1, v2; cor12 = r_per_study) # one per study
meta_vcov_bivariate(v1, v2; cov12 = c_per_study) # covariances directly
```

At most one of `cor12` / `cov12`; the default is independence. Each 2×2 block
must be a valid covariance matrix (``c_i^2 \le v_{1i}v_{2i}``) and is checked at
construction, so an impossible input fails immediately rather than as a strange
optimisation failure later.

## 3. Coming from R

drmTMB writes the bivariate case as

```r
drmTMB(bf(mu1 = y1 ~ x + meta_V(V = V), mu2 = y2 ~ x,
          sigma1 = ~1, sigma2 = ~1, rho12 = ~1),
       family = c(gaussian(), gaussian()), data = dat)
```

DRModels.jl takes the object as a **fit-call keyword** instead:

```julia
drm(bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x), …),
    Gaussian(); data = dat, V = V)
```

This is a deliberate divergence, and the reason is mechanical rather than a
preference: `meta_V(V = …)` puts a **keyword argument inside a formula**, which
Julia's `@formula` macro cannot represent at all. So `V` sits with `tree`, `K`
and `A` — the other "structure supplied alongside the data" arguments. Writing
`meta_V(...)` inside a bivariate formula errors with a pointer to the keyword.

`Matrix(V)` produces drmTMB's dense ``2n \times 2n`` block-diagonal form
(stacking order ``y_1[1], y_2[1], y_1[2], \dots``), and that matrix is accepted
back directly, so a `V` built in R can be handed over unchanged.

### How wrong can your `cor12` guess be?

Often you know `v1` and `v2` exactly but must *estimate* the sampling
correlation. In a 40-replicate recovery study with true sampling correlation
0.6 and true heterogeneity correlation ρ = −0.35:

| assumed `cor12` | bias in `rho12` |
|---|---|
| 0.0 | +0.062 |
| 0.3 | +0.030 |
| **0.6** (correct) | −0.003 |
| 0.9 | −0.035 |

The bias is linear in the error of the guess, at roughly

```
bias in rho12  ≈  -0.10 × (assumed cor12 − true cor12)
```

— a **10:1 attenuation**. Being wrong by 0.3 costs about 0.03. So a
plausible-but-imperfect `cor12` is far better than none, and it is worth
reporting which value you assumed.

**And if you have no idea at all, still pass `V` with `cor12 = 0`** rather than
omitting `V`: supplying the sampling *variances* matters separately from the
correlation, and omitting `V` entirely was measurably worse (+0.091) than
assuming independence (+0.062). This comparison comes from the package's
parameter-recovery checks for the bivariate meta-analysis route.

## 4. What is *not* covered

- **Known covariance with structured effects.** `V` combined with
  `phylo`/`relmat`/`animal`/`spatial` markers on the bivariate route is refused;
  phylogenetic meta-analysis with known sampling covariance is not currently
  supported.
- **REML with `V`** — ML only.
- **Cross-study sampling covariance.** Only row-paired 2×2 blocks are supported.
  A dense matrix carrying off-block entries is **refused**, not silently
  truncated, because dropping them would fit a different model than the matrix
  describes.
- **Univariate `meta_V` remains diagonal.** The bivariate path is the dense one.

## 5. Verifying your own fit

The heterogeneity split is exactly where a meta-analysis goes wrong quietly, so
it is worth one check: fit with and without `V` and confirm `rho12` moves.

```julia
with    = drm(bfbiv, Gaussian(); data = dat, V = V)
without = drm(bfbiv, Gaussian(); data = dat)
with.scales[:rho12][1], without.scales[:rho12][1]
```

If they are identical, `V` is not reaching the likelihood — check that its length
matches the number of studies.

## See also

- [`meta_vcov_bivariate`](@ref) — the constructor and its validation.
- [Meta-analysis tutorial](../tutorials/meta-analysis.md) — worked univariate
  and multilevel examples.
- The R↔Julia phrasebook: [`rosetta.md`](../rosetta.md).
