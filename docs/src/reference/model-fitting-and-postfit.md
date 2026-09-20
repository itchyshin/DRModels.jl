# Model fitting and post-fit tools

!!! note "Status — Reference"
    Mirrors drmTMB's [Model fitting and post-fit tools](https://itchyshin.github.io/drmTMB/reference/index.html) (23 in drmTMB). The fitting verb is [`drm`](@ref); the post-fit accessors below cover coefficients, fitted scale / correlation, random-effect estimates, predictions, simulation, inference, and convergence diagnostics.

## Fitting

```@docs
drm
DrmFit
```

## Coefficients and (co)variance

```@docs
fixef
re_sd
vc
coeftable
coef
vcov
nobs
```

## Fitted scale and correlation

```@docs
sigma
corpairs
rho12
coevolution_cor
coevolution_vc
coevolution_summary
```

## Random-effect estimates

```@docs
ranef
```

## Prediction and simulation

```@docs
predict
predict_parameters
marginal_parameters
prediction_grid
simulate
fitted
residuals
```

### Predicting distributional parameters

!!! warning "Prediction interpretation"
    The embedded `predict_parameters` docstring above uses legacy wording about
    integrating effects out. The current implementation sets random and
    structured effects to zero. With a nonlinear link, these are different
    predictions: for example, `exp(η)` differs from averaging `exp(η + b)` over
    a non-degenerate random effect `b`. Use the fixed-effect interpretation here.

[`predict`](@ref) returns the response (mean) prediction, but a distributional
regression also models the scale and—bivariately—the correlation. Use
[`predict_parameters`](@ref) to obtain the population-level value of **every**
distributional parameter the model carries (`:mu`, `:sigma`, plus any family
extras) at new covariate values, with random / structured effects set to zero.
This is an inverse-link transformation of the fixed-effect linear predictor,
not response-scale integration over the random-effect distribution. [`marginal_parameters`](@ref) is the cheap in-sample accessor that reads the
fitted per-observation parameters straight off the fit. [`prediction_grid`](@ref)
builds the new-data table to sweep over (varying chosen predictors, holding the
rest at a reference value).

```julia
using DRModels, Random
Random.seed!(20260603)

x = randn(500)
y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(500)
data = (; y, x)

# Gaussian location–scale fit: both μ and σ depend on x.
fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

# Per-distributional-parameter prediction over a covariate sweep.
grid = prediction_grid((; x = data.x), x = range(-2, 2; length = 11))
p = predict_parameters(fit, grid)   # Dict(:mu => …, :sigma => …) on the response scale
p[:mu]                              # predicted mean across the sweep
p[:sigma]                          # predicted scale across the sweep

# Working (link) scale instead: returns Xβ̂ per parameter.
predict_parameters(fit, grid; type = :link)[:mu]

# In-sample fitted parameters, straight off the fit (no recomputation).
marginal_parameters(fit)            # == predict_parameters(fit, data) in-sample
```

## Formula-fitted finite-state missing predictors

!!! warning "Experimental"
    Exported for evaluation, not yet stable. API and numerics may
    change. This route is not available through the R bridge.

For one ordinal or categorical missing predictor in the bounded Gaussian joint
route, construct the predictor model with `impute_model`; use
`CategoricalLogit()` for nominal states. `JointFiniteDrmFit` retains the raw
kernel coefficients and covariance. For ordinal predictors, `cutpoints(fit)`
provides the constrained cutpoints separately. See the engine-internals
reference for the prepared-state design and route limits.

```@docs
DRModels.CategoricalLogit
DRModels.JointFiniteDrmFit
DRModels.cutpoints
```

## Inference

Check the status as well as the bounds of a profile interval. A signed infinite
bound can mean that no crossing was found within the searched range, or that an
endpoint solve failed. `profile_result` distinguishes these outcomes in `stats`
and `failed`; `confint` warns about failed endpoint solves.

For coupled non-Gaussian location–scale fits, `endpoint_diagnostics` also records
why each endpoint search stopped, its last evaluated candidate and its residual.
A candidate from a failed search is diagnostic information, not a confidence
limit. Through the experimental R bridge (`engine = "julia"`), inspect
`conf.status` and `profile.message` in the returned interval table. A failed result must not be read as evidence of
an unbounded interval.

```@docs
confint
stderror
profile_result
bootstrap_ci
bootstrap_summary
bootstrap_result
```

## Information criteria

```@docs
loglik
ml_loglik
reml_loglik
estimation_method
reml_objective_at
dof
aic
bic
aicc
```

## Model comparison

```@docs
lrtest
anova
weights
update
```

## Diagnostics and accessors

```@docs
check_drm
family
is_converged
deviance
dof_residual
r2_constant_sigma
```

## Fit summaries and route inventories

```@docs
Base.summary(::DrmFit)
niterations
profile_targets
structured_effects
```

## Derived phylogenetic quantities and boundary comparisons

```@docs
gaussian_locscale_phylo_sds
profile_sigma_a
bootstrap_sigma_a
repeatability
lrt_boundary
```

## R bridge and preprocessing

These entries document the Julia-side interface used by the optional R bridge.
[Coming from R](../coming-from-r.md) explains the optional route and
refusals; these docstrings do not expand that contract.

```@docs
drm_bridge
drm_bridge_inference
drm_bridge_objective_at
drm_listwise
```

## Staged pair association diagnostics

```@docs
PairAssociation
integration_diagnostics
```

## Phylogenetic penalty (MAP)

[`drm_phylo_penalty`](@ref) is the Julia twin of drmTMB's `drm_phylo_penalty()`:
pass `penalty = drm_phylo_penalty(...)` to [`drm`](@ref) to turn a phylogenetic
variance-component fit into a **MAP** estimate. [`drm_phylo_penalty_sweep`](@ref)
refits across a grid of correlation-penalty scales so you can see whether a
conclusion moves with the prior. [`PhyloCorPenaltyNeedsTwoSD`](@ref) is the
typed refusal when `cor_sd` is asked of a model that has no phylogenetic
correlation to penalize.

```@docs
drm_phylo_penalty
drm_phylo_penalty_sweep
PhyloPenalty
PhyloCorPenaltyNeedsTwoSD
```

## Staged pair association

[`associate_pairs`](@ref) estimates a latent-normal association between two
already-fitted univariate models (drmTMB's staged, frozen-margin route). The
kernel must be given explicitly as [`latent_normal`](@ref); [`association`](@ref)
is the post-fit summary.

```@docs
associate_pairs
latent_normal
association
```

## Heritability, ICC, and derived quantities

```@docs
heritability
icc
bias_correct
```

## Boundary inference

```@docs
chibar_pvalue
```

## Cross-family post-fit

Accessors for a `fit_mixed_family` result. The cross-family bivariate route is
**experimental** and not ready for routine use: narrow documented evidence, no interval
coverage, and the dependence it reports is a latent-scale scalar correlation
(`fit.rho_latent`), not a `rho12` formula. [`mf_coef`](@ref) is the tidy
coefficient table; the other `mf_*` helpers live beside it in the module.

## Engine constructors (q=4 and coevolution)

`AugProblem`, `make_problem`, and `fit_q4_sparse_tmb` are low-level exported
interfaces behind the q=4 PLSM path. Use `drm(...)` for ordinary model fitting;
these bindings serve scripts that prepare engine inputs directly.

### [AugProblem](@id AugProblem)

`AugProblem` holds the augmented-phylogeny data and q=4 design matrices used
by the sparse engine.

### [make_problem](@id make_problem)

`make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr; species = 1:phy.n_leaves)`
builds that problem and its root-conditioned precision from a phylogeny.

### [fit_q4_sparse_tmb](@id fit_q4_sparse_tmb)

`fit_q4_sparse_tmb(prob, Q_cond; θ0 = ..., ...)` runs sparse q=4 optimisation.
Starting values must be supplied as either `θ0` (the full parameter vector)
or `β0` (the mean coefficients).

The documented q=4 marginal evaluator and general-q coevolution bindings are
listed below.

```@docs
marginal_nll
CoevoProblem
lc_to_cov
```
