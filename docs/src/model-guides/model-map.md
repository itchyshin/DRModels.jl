# What can I fit today?

Use this page when you know what you measured but are deciding what kind of model
to fit. DRModels.jl is for questions where predictors may change not only the
average response, but also its variability, its zeros, or the way two responses
vary together.

Start with the question closest to yours:

| Your response or question | A good first page |
|---|---|
| A continuous measurement, such as a trait, concentration, or score | [Fit your first model](../getting-started.md) |
| Counts, including overdispersed counts or extra zeros | [Count abundance and extra zeros](../tutorials/count-nbinom2.md) |
| Proportions, success rates, or binomial data | [Proportions and success rates](../tutorials/proportion-beta-binomial.md) |
| Whether predictors change both the average and the spread | [When variance carries signal](../tutorials/location-scale.md) |
| Two responses that may be associated | [Changing residual coupling](../tutorials/bivariate-coscale.md) |
| Observations linked by a phylogeny, space, pedigree, or relatedness matrix | [Biological examples](../tutorials/phylogenetic-models.md) |

The first examples are motivated by ecology, evolution, and environmental
science, but the models are also useful wherever a response can vary in more
than its average. Read on for the model choices, or use the
[which page next](#Which-page-next) table to move directly to a guide.

## The modelling idea: a formula per parameter

DRModels.jl is **distributional regression** — it lets you model more than the
mean of a response. Each parameter supported by the chosen response family can
have its own formula, bundled together with [`bf`](@ref):

| Parameter | What it controls | Formula |
|---|---|---|
| **μ** (mean / location) | where the response sits | the response formula, `y ~ …` |
| **`sigma`** (scale / dispersion) | how spread out it is | `sigma ~ …` |
| family extras — `nu`, `zi`, `hu`, `zoi`, `coi` | shape, zero-inflation, hurdle, boundary mass | `nu ~ …`, `zi ~ …`, … |
| **`rho12`** (bivariate only) | residual correlation between two responses | `rho12 ~ …` |

Always write `sigma` (never `tau`) for the scale and `rho12` for residual
correlation. The same machinery drives every parameter: a formula → a linear
predictor → a family-specific link. Which parameters are *available* depends on
the family (Gaussian has no `nu`; only counts take `zi` / `hu`).

See [Which scale are you modelling?](which-scale.md) for the difference between
the residual `sigma`, a group-level SD, and a known sampling variance — they are
distinct quantities DRModels.jl keeps separate.

## The `bf(...)` front end

[`bf`](@ref) (alias `drm_formula`) collects one formula per parameter, following
the same formula-per-parameter pattern as drmTMB and brms. It has two forms:

**Univariate — positional.** The first formula is the mean; the rest name their
parameter on the left-hand side:

```julia
bf(@formula(y ~ x), @formula(sigma ~ x))                 # mean + scale
bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))   # + shape (Student)
```

**Bivariate — keyword.** Name each of the two responses' parameters explicitly,
including the residual correlation `rho12`:

```julia
bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
   rho12 = @formula(rho12 ~ x))
```

You pass the bundle and a family to [`drm`](@ref):
`drm(bf(...), Gaussian(); data = dat)`. ML is the default (REML likelihoods are
not comparable across fixed-effect structures, so ML is what model selection
needs). For the full grammar — including `cbind(successes, failures) ~ …` for
binomial-type responses — see the
[model specification reference](../reference/model-specification.md).

## Supported families

Pick the family from the *shape* of the response; the
[Choosing response families](distribution-families.md) guide has the full
decision table and worked examples. DRModels.jl implements drmTMB's complete family
set:

| Family | Response | Mean link | `sigma` slot / extras |
|---|---|---|---|
| [`Gaussian`](@ref) | real-valued, symmetric | identity | residual SD `σ` (log) |
| [`Student`](@ref) | real-valued, heavy tails | identity | scale `σ` + d.o.f. `nu` |
| [`LogNormal`](@ref) | positive, multiplicative | identity on `log y` | SD of `log y` |
| [`Gamma`](@ref) | positive, continuous | log | CV → shape `α = 1/σ²` |
| [`Tweedie`](@ref) | positive **with exact zeros** | log | √dispersion `σ` + power `nu` ∈ (1,2) |
| [`Poisson`](@ref) | counts (var ≈ mean) | log | — (`+ zi` / `+ hu`) |
| [`NegBinomial2`](@ref) | overdispersed counts (NB2) | log | dispersion `θ` (`+ zi` / `+ hu`) |
| [`TruncatedNegBinomial2`](@ref) | positive counts (≥ 1) | log | dispersion `θ` |
| [`Beta`](@ref) | proportions in (0,1) | logit | precision `φ = 1/σ²` |
| [`BetaBinomial`](@ref) | successes / trials, overdispersed | logit | `φ = 1/σ²` (`cbind`) |
| [`Binomial`](@ref) | successes / trials | logit | — (`cbind` or 0/1) |
| [`ZeroOneBeta`](@ref) | proportions on `[0,1]` incl. 0 and 1 | logit | `φ` + `zoi` / `coi` |
| [`CumulativeLogit`](@ref) | ordered categories `1..K` | logit (cutpoints) | K−1 ordered cutpoints |

The count modifiers — `zi` (zero-inflation, ZIP / ZINB), `hu` (hurdle), and the
beta boundary modifiers `zoi` / `coi` — are themselves formulas you add to the
bundle, e.g. `bf(@formula(y ~ x), @formula(zi ~ 1))`.

## Structured and random effects

On top of fixed effects, DRModels.jl carries ordinary random effects and several
**structured** effects whose covariance comes from a known matrix or geometry.
Write them as terms in the mean formula:

| Effect | Term | What it encodes |
|---|---|---|
| Random intercept / slope | `(1 \| g)`, `(0 + x \| g)`, `(1 + x \| g)` | exchangeable group variation; correlated slopes |
| Crossed / nested RE | `(1 \| g) + (1 \| h)` | multiple grouping factors |
| [`phylo`](@ref) | `phylo(1 \| species)` | covariance from a phylogenetic tree |
| [`spatial`](@ref) | `spatial(1 \| site)` | exponential kernel over coordinates, estimated range |
| [`animal`](@ref) | `animal(1 \| id)` | additive-genetic covariance from a pedigree `A` |
| [`relmat`](@ref) | `relmat(1 \| id)` | a user-supplied relatedness matrix `K` |
| [`meta_V`](@ref) | `meta_V(v)` | known sampling (co)variances for meta-analysis |

**Which families route which effects.** Gaussian gets the full set — random
intercept/slope, correlated and crossed RE, phylo / spatial / animal / relmat
structure, `meta_V`, and a random effect *on* `sigma` — fit in closed form or
via a sparse augmented-state Laplace approximation. The non-Gaussian families
(Poisson, NB2, Binomial, Gamma, Beta, Student-t, LogNormal, Beta-binomial,
Tweedie and CumulativeLogit) carry random intercepts via Gauss–Hermite
marginals, and random slopes for every one of them except `Binomial()`, which
supports `(1 | g)` on the mean only. The [capability matrix](../capabilities.md) records which slope form —
independent or correlated — each family admits.
**Phylogenetic** (`phylo`) effects go via a sparse Laplace path for Poisson,
NB2, Binomial, Gamma, Beta, Beta-binomial and CumulativeLogit. On that path NB2, Gamma and Beta
accept a covariate dispersion formula `sigma ~ x` (a per-observation log σ),
while `BetaBinomial()` requires a constant `sigma` and `Binomial()`
carries no dispersion parameter at all; `Student()` rejects `meta_V` and every structured marker. `LogNormal()` is the
exception: because `log y` is exactly Gaussian, its `phylo`/`relmat`
structured markers on the mean delegate WHOLESALE to `Gaussian()` on
`log y` (exact, not a Laplace approximation) rather than the shared
non-Gaussian sparse path; `animal`/`spatial` are not implemented for
`LogNormal()`. So put predictors on `sigma` for Gaussian freely. On the
non-Gaussian phylogenetic path, follow the family-specific boundary: NB2,
Gamma, and Beta accept `sigma ~ x`; `BetaBinomial()` requires constant
dispersion; and `Binomial()` has no dispersion formula.

[`CumulativeLogit`](@ref) (ordinal) carries an ordinary random intercept
`(1 | g)` or an *independent* random slope `(0 + x | g)` on `mu` via the same
Gauss–Hermite scheme, and an intercept-only `phylo(1 | species)` through the
same sparse Laplace engine; the correlated form `(1 + x | g)` and the other
structured effects (`relmat` / `animal` / `spatial`) are not implemented yet.

For the verified engine behind the phylogenetic models — the q=4 phylogenetic
bivariate location–scale model, which matches drmTMB's fit and still returns
usable Wald and bootstrap intervals where drmTMB's Hessian is singular — see
[Large data](large-data.md) for the scoped performance evidence.

## After the fit

The common starting points are [`coef`](@ref), `summary` / [`coeftable`](@ref),
[`fitted`](@ref), response [`residuals`](@ref), and [`aic`](@ref) / [`bic`](@ref)
for appropriate ML comparisons. The rest depends on the fitted route:
standard errors and Wald intervals require an available covariance estimate;
profile intervals require a stored or precomputed profile target; and bootstrap,
prediction, simulation, and variance-component summaries apply only where that
model implements them. Use `profile_targets(fit)` to see which profile intervals
are ready for a particular fit, and consult the [capability matrix](../capabilities.md)
before planning an analysis around a post-fit method. [`re_sd`](@ref) and
[`vc`](@ref) describe mixed or structured models, not every fit.

See [Checking and using fitted models](model-workflow.md) for a worked Gaussian
workflow, and [Which scale are you modelling?](which-scale.md) for honest
inference at a variance boundary.

## Which page next

| If you want to… | Go to |
|---|---|
| Fit your first model, end to end | [Get started](../getting-started.md) |
| Choose the right response family | [Choosing response families](distribution-families.md) |
| Tell residual `σ`, group SD, and known V apart | [Which scale are you modelling?](which-scale.md) |
| Extract coefficients, CIs, predictions | [Checking and using fitted models](model-workflow.md) |
| Diagnose convergence | [Convergence](convergence.md) |
| Scale to large data | [Large data](large-data.md) |
| Choose the marginal method (Laplace vs VA) | [Marginal: LA vs VA](marginal-la-vs-va.md) |
| Model variability as signal (location–scale) | [When variance carries signal](../tutorials/location-scale.md) |
| Predictors on a random effect's SD (location–scale–scale, incl. climate-dependent phylogenetic SD) | [Part 2: location–scale–scale](../tutorials/location-scale-scale.md) |
| Change residual coupling with `rho12` | [Changing residual coupling with rho12](../tutorials/bivariate-coscale.md) |
| Robust continuous responses | [Robust continuous responses](../tutorials/robust-student.md) |
| Counts and extra zeros | [Count abundance and extra zeros](../tutorials/count-nbinom2.md) |
| Proportions and success rates | [Proportions and success rates](../tutorials/proportion-beta-binomial.md) |
| Phylogenetic / spatial / animal models | [Phylogenetic](../tutorials/phylogenetic-models.md) · [Spatial](../tutorials/spatial-models.md) · [Animal](../tutorials/animal-models.md) |
| Meta-analysis with known variances | [Meta-analysis](../tutorials/meta-analysis.md) |
| The full API reference | [Model specification](../reference/model-specification.md) · [Fitting & post-fit](../reference/model-fitting-and-postfit.md) |
