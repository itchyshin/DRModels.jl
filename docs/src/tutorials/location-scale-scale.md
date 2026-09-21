# When variance carries signal, Part 2: location–scale–scale

!!! note "Status — Experimental"
    Mirrors drmTMB's [location-scale-scale](https://itchyshin.github.io/drmTMB/articles/location-scale-scale.html)
    vignette. **In DRModels.jl today:** `sd(group) ~ z` on the iid `(1 | g)` random
    effect (ML + REML), `sd(group, phylogenetic) ~ z` on the per-species
    phylogenetic SD (ML + REML, with dense and $O(p)$ sparse solvers), and
    multi-component LSS models. Both are Experimental-tier
    ([API stability](../api-stability.md)). The examples below use identical
    data in DRModels.jl and drmTMB when comparing numerical results.

A [location–scale model](location-scale.md) asks whether predictors change the
expected response `μ` and the residual SD `σ`. A **location–scale–scale** model
adds a third submodel: predictors can also change the standard deviation of a
**latent random effect**. DRModels.jl writes that third submodel as `sd(group) ~ z`,
exactly as drmTMB.

## Personality, predictability, repeatability

Suppose an exploration score is recorded repeatedly for each individual, and sex
may predict three different things at once:

1. the **mean** score;
2. **between-individual** variation (how much individuals differ from each other);
3. **within-individual** variation (how repeatable each individual is).

For observation `j` of individual `i`:

```math
\begin{aligned}
y_{ij} &\sim \mathrm{N}(\mu_{ij},\, \sigma_{e,i}^2), &
\mu_{ij} &= \beta_0 + \beta_1\,\mathrm{sex}_i + b_i,\\
b_i &\sim \mathrm{N}(0,\, \sigma_{b,i}^2), &
\log \sigma_{e,i} &= \gamma_0 + \gamma_1\,\mathrm{sex}_i,\\
&& \log \sigma_{b,i} &= \alpha_0 + \alpha_1\,\mathrm{sex}_i.
\end{aligned}
```

The matching formula bundle — one formula per submodel:

```@example lss
using DRModels, Random

rng = Random.MersenneTwister(20260715)
n_id, n_each = 80, 6
sex = repeat([0.0, 1.0], inner = n_id ÷ 2)                 # 0 = female, 1 = male
b = randn(rng, n_id) .* [0.65, 0.40][Int.(sex) .+ 1]       # between-SD differs by sex
id = repeat(1:n_id, inner = n_each)
sexl = sex[id]
y = [0.35, 0.70][Int.(sexl) .+ 1] .+ b[id] .+
    randn(rng, n_id * n_each) .* [0.35, 0.60][Int.(sexl) .+ 1]
dat = (; y, sex = sexl, id)

fit = drm(bf(@formula(y ~ sex + (1 | id)),
             @formula(sigma ~ sex),
             @formula(sd(id) ~ sex)), Gaussian(); data = dat)

coef(fit, :sd)          # log between-individual SD: (Intercept), sex
```

The same predictor appears in three formulas, and its three coefficients answer
three separate questions. All SD submodels use a log link, so read them back
through `exp`. On this simulation (truth: between-SD 0.65 vs 0.40, within-SD
0.35 vs 0.60):

```@example lss
(between_sd = exp.([1 0; 1 1] * coef(fit, :sd)),        # ≈ 0.65 (F), 0.40 (M)
 within_sd  = exp.([1 0; 1 1] * coef(fit, :sigma)))     # ≈ 0.35 (F), 0.60 (M)
```

!!! warning "sd() predictors must be constant within each group"
    Sex is used to model `sd(id)`, so it must not vary within an individual.
    DRModels.jl checks this and errors, naming the offending predictor — exactly as
    drmTMB does. Do not average a genuinely within-group predictor to silence
    the error; that changes the scientific question.

Because the RE SD now varies by group, single-number summaries are ill-defined
and refuse rather than misreport: `re_sd`, `vc`, and `heritability` all throw
with a pointer to `coef(fit, :sd)`. Wald and profile intervals target the block
as usual:

```@example lss
confint(fit; parm = :sd)
```

`method = :REML` is supported on this route.

## The phylogenetic scale: `sd(species, phylogenetic)`

For comparative data the deeper question is usually about the **phylogenetic**
SD: does among-species variation that tracks the tree change along an
environmental gradient? With one trait value per species `i`:

```math
y_i \sim \mathrm{N}\!\left(\mu_i,\; \sigma_{a,i}^2 + \sigma_{e,i}^2\right),
\qquad
\mathbf{a} \sim \mathrm{MVN}\!\left(\mathbf{0},\, D_a\,A\,D_a\right),
```

where `A` is the Brownian-motion phylogenetic correlation,
`D_a = \mathrm{Diagonal}(\sigma_{a,1},\dots,\sigma_{a,n})`, and each of
`log σ_a` and `log σ_e` carries its own linear predictor. This is the
location–scale–scale framework of the Mizuno et al. ecogeographical-rules
protocol: climate in the mean, the non-phylogenetic scale, **and** the
phylogenetic scale.

```@example lss
using LinearAlgebra
# a balanced 64-tip unit-height tree
function _baln(d)
    node(p, k) = k == 0 ? "$(p):$(1/d)" :
        (k == d ? "($(node(p*"a",k-1)),$(node(p*"b",k-1)));" :
                  "($(node(p*"a",k-1)),$(node(p*"b",k-1))):$(1/d)")
    node("t", d)
end
phy = DRModels.augmented_phy(_baln(6))
G = phy.n_leaves
K0 = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
dK = sqrt.(diag(K0)); K = K0 ./ (dK * dK')

rng2 = Random.MersenneTwister(11)
x = randn(rng2, G)                            # standardised "climate"
sda = exp.(-0.5 .+ 0.4 .* x)                  # phylo SD rises with climate
sde = exp.(-1.0 .- 0.3 .* x)                  # residual SD falls with climate
a = Diagonal(sda) * (cholesky(Symmetric(K)).L * randn(rng2, G))
yq = 1.0 .+ 0.5 .* x .+ a .+ sde .* randn(rng2, G)
datq = (y = yq, x = x, species = String.(phy.leaf_names))

fitq = drm(bf(@formula(y ~ x + phylo(1 | species)),
              @formula(sigma ~ x),
              @formula(sd(species, phylogenetic) ~ x)),
           Gaussian(); data = datq, tree = phy)

(mu = coef(fitq, :mu), sigma = coef(fitq, :sigma), sd_phylo = coef(fitq, :sd_phylo))
```

The estimates track the simulated truth (mean 1.0 and 0.5; the σ and σ_a
slopes in the right directions). For these same data, drmTMB returns the same
log-likelihood (−69.1373) and the same coefficients to seven significant
figures.

Species rows need not follow tree-tip order when fitting an LSS model. String
labels match `phy.leaf_names` exactly; integer labels are positions in `1:G`,
not arbitrary group identifiers. Every tip must be represented in the full
input, even if all responses for a tip are missing. Scale predictors must still
be constant within each species.

```@example lss
row_order = randperm(rng2, G)
datq_shuffled = (; y = datq.y[row_order], x = datq.x[row_order],
                  species = datq.species[row_order])
fitq_shuffled = drm(bf(@formula(y ~ x + phylo(1 | species)),
                      @formula(sigma ~ x),
                      @formula(sd(species, phylogenetic) ~ x)),
                   Gaussian(); data = datq_shuffled, tree = phy)
@assert isapprox(loglik(fitq_shuffled), loglik(fitq); atol = 1e-7, rtol = 0)
@assert isapprox(coef(fitq_shuffled, :sd_phylo), coef(fitq, :sd_phylo);
                 atol = 4e-6, rtol = 0)
coef(fitq_shuffled, :sd_phylo)
```

!!! note "Bootstrap draws follow the fitted variance model"
    For Gaussian models with `sd()` submodels, each bootstrap replicate redraws
    every IID and phylogenetic mean effect independently, using tree-tip names
    to match species. It also redraws residual error. Missing responses remain
    missing in each refit, and a REML seed fit is refitted with REML.
    These simulation checks do not establish interval coverage or large-tree
    performance; inspect failed-refit counts before interpreting intervals.

!!! note "Grammar and solver notes"
    - `sd(species, phylogenetic)` is the canonical spelling (drmTMB:
      `sd(species, level = "phylogenetic")`; `@formula` cannot parse keyword
      arguments, so the level is a bare symbol). The legacy `sd_phylo(species)`
      still works but is soft-deprecated on both sides.
    - The mean formula must carry the matching `phylo(1 | species)` marker, and
      `tree = …` is required.
    - **ML and REML**: both `method = :ML` (default) and `method = :REML` are
      supported on phylogenetic and iid LSS routes.
    - **Sparse whole-tree scaling**: for large trees, `sparse = true` (or
      `algorithm = :sparse_lbfgs`) invokes the exact $O(p)$ augmented-state
      GMRF engine with Takahashi selected-inverse leaf variances, avoiding
      $O(G^2)$ dense matrix storage and $O(G^3)$ dense factorization. This sparse engine is selected automatically
      when $G > 500$ species.

## Missing response handling

Like other Gaussian routes in DRModels.jl, Location–Scale–Scale models support
incomplete responses (`missing` or `NaN` in `y`), matching `response = "include"`
in drmTMB:

```@example lss
# Mask some responses
y_miss = Vector{Union{Float64, Missing}}(copy(dat.y))
y_miss[1:5] .= missing
dat_miss = (; y = y_miss, sex = dat.sex, id = dat.id)

fit_miss = drm(bf(@formula(y ~ sex + (1 | id)),
                  @formula(sigma ~ sex),
                  @formula(sd(id) ~ sex)),
               Gaussian(); data = dat_miss)
nobs(fit_miss)   # count of observed rows
```

Group-level designs ($Z_g$, $D_a$, $G$) are constructed from the full grouping
data so the random-effect scale is parameterised across all levels, while the
likelihood is evaluated over observed rows.

## REML on location–scale–scale models

REML accounts for estimating fixed effects when fitting covariance parameters:

```@example lss
fit_reml = drm(bf(@formula(y ~ sex + (1 | id)),
                  @formula(sigma ~ sex),
                  @formula(sd(id) ~ sex)),
               Gaussian(); data = dat, method = :REML)

reml_loglik(fit_reml)   # restricted log-likelihood
```

Both iid and phylogenetic LSS models support REML estimation. Standard errors
can be unreliable when a variance approaches zero; a successful fit alone does
not establish reliable uncertainty estimates.

To assess the bootstrap rather than only its interval endpoints, retain the
full result:

```@example lss
boot = bootstrap_result(fit_reml; data = dat, B = 4,
                        rng = MersenneTwister(20260830), threads = true,
                        failures = :skip, check_converged = true)
@assert boot.attempted == boot.used + boot.failed
(attempted = boot.attempted, used = boot.used, failed = boot.failed)
```

Four replicates keep this example quick; they are too few for an interval you
would report. Choose the number of replicates for your analysis and retain
`boot.failures`, which records failed replicate seeds and messages. For a
phylogenetic model, also pass the original `tree`. For profiles, inspect
`profile_result(...).stats`: the generic path records nuisance termination and
failed endpoints separately from limits of the searched range. These checks
do not establish interval coverage or practical performance on large trees.

## From R, with intervals

The same models run from R through `engine = "julia"`, with Wald, profile, and
parametric-bootstrap intervals:

```r
fit <- drmTMB(
  bf(y ~ temp + I(temp^2) + prec + I(prec^2) + phylo(1 | species, tree = tr),
     sigma ~ temp + prec,
     sd(species, level = "phylogenetic") ~ temp + prec),     # the "M6" shape
  family = gaussian(), data = d, engine = "julia"
)
confint(fit, parm = "fixef:sd_phylo:temp", method = "profile")
confint(fit, parm = "fixef:sd_phylo:temp", method = "bootstrap", R = 199,
        threads = TRUE)          # threaded refits
```

For the ecogeographical-rules formulas assessed here, the Julia and TMB fits
gave identical log likelihoods.

## See also

- [When variance carries signal](location-scale.md) — Part 1: `μ` and `σ`.
- [Phylogenetic structured effects](phylogenetic-models.md) — the `phylo()`
  marker and tree handling.
- [API stability](../api-stability.md) — `sd` / `sd_phylo` sit on the
  Experimental tier while the surface settles.
