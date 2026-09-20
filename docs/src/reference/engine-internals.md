# Engine internals and advanced constructors

!!! warning "Developer reference"
    This appendix records docstrings for the sparse, coevolution, bridge, and
    marginal-likelihood implementation surface. Names beginning with `_` are
    nonpublic implementation helpers. Exported advanced constructors are
    included for source coverage, not as a claim that every combination is a
    supported `drm(...)` route. Start with the model guides and capability
    matrix for ordinary fitting.

## Marginal-method and association internals

```@docs
DRModels.AGHQ
DRModels.Laplace
DRModels.Variational
DRModels.MarginalMethod
DRModels.LatentNormal
DRModels._aghq_1d_logint
DRModels._aghq_require_1d
DRModels._assoc_logdiffexp
DRModels._poisson_group_aghq_logint
```

### Profile nuisance-status internals (no stability guarantee)

```@docs
DRModels._ProfileNuisanceResult
```

## Augmented phylogeny and sparse linear algebra

```@docs
DRModels.AugmentedPhy
DRModels.Vinv_mul
DRModels.build_M
DRModels.exact_traces
DRModels.logdetV_val
DRModels.make_phy
DRModels.make_problem_from_Q
DRModels.q4_marginal_diagnostic
DRModels.takahashi_diag
DRModels.takahashi_selinv
DRModels.theta_len
DRModels.unpack_theta
DRModels.pack_theta
```

### Cholesky pattern-reuse internals (no stability guarantee)

Opt-in `cholesky!` symbolic reuse on the q=4 phylo ML route (`chol_ref`
kwarg; the default `nothing` keeps a fresh factorisation per call).

```@docs
DRModels.CholPatternCache
DRModels.CholPatternMismatch
DRModels._assert_chol_pattern_matches
DRModels._add_diag
```

### Newick and topology parser internals (no stability guarantee)

```@docs
DRModels._phy_validate_topology
DRModels._parse_label!
```

## q=4 and D·R·D separation evaluators

The `fz_` prefix and `fit_q4_sparse_fisherz` are historical names: the
correlation map is spherical/LKJ, not Fisher-z. The warning on
`DRModels.fit_q4_sparse_fisherz` below gives the correct back-transform.

```@docs
DRModels.beta_widths
DRModels.fit_q4_reml
DRModels.fit_q4_sparse_fisherz
DRModels.fz_DRD
DRModels.fz_R
DRModels.fz_R_chol
DRModels.fz_init_from_Sigma
DRModels.fz_marginal_and_grad
DRModels.fz_marginal_nll
DRModels.fz_psi_len
DRModels.fz_psi_to_theta
DRModels.fz_unpack_psi
DRModels.marginal_and_exact_grad
```

## Coevolution and phylogenetic interaction kernels

```@docs
DRModels.coevo_marginal_cov
DRModels.coevo_pack
DRModels.coevo_unpack
DRModels.cov_to_lc
DRModels.fit_coevolution
DRModels.fit_coevolution_q2_reml
DRModels.fit_coevolution_q2_residual
DRModels.fit_phylo_interaction
DRModels.lc_len
DRModels.lc_metric
DRModels.make_coevo_problem
DRModels.make_coevo_problem_from_covariance
DRModels.make_coevo_problem_from_precision
DRModels.phylo_interaction_nll
DRModels.simulate_coevolution
```

## Bridge and mixed-family payload helpers

```@docs
DRModels._bridge_dpars
DRModels._bridge_dpars_newdata
DRModels._bridge_meta_parts
DRModels._bridge_trials
DRModels._bridge_mixture_family_hint
DRModels._bridge_fitted_marginal
DRModels.drm_bridge_q2_known_precision
DRModels.drm_bridge_q2_phylo
DRModels.fit_mixed_family
DRModels._mf_nparams
```

### Bridge label metadata internals (no stability guarantee)

```@docs
DRModels._BridgeFormulaLabels
```

## Location-scale and structured-fit kernels

```@docs
DRModels._fit_bivariate_q4_structured
DRModels._fit_cumulative_phylo_laplace
DRModels._fit_corr_locscale
DRModels._fit_fixed_gaussian_reml
DRModels._fit_gaussian_locscale_phylo
DRModels._fit_locscale
DRModels._fit_ranef_gaussian
DRModels._fit_sigma_axis_re
DRModels._general_cov_setup
DRModels._ls_components
DRModels._ls_marginal_grad
DRModels._ls_marginal_nll
DRModels._ls_obs_information
DRModels._ls_profile_ci
DRModels._ls_profile_ci_result
DRModels._phylo_mean_laplace_hetero_fg
DRModels._phylo_mean_leaf_index
DRModels._reml_normalise
DRModels._vcov_from_hessian
DRModels.q2_reml_phi_len
```

### Paired whitening internals (no stability guarantee)

These private helpers implement the transformed latent-state route used by
coupled non-Gaussian location-scale fitting, observed information, and marginal
bootstrap simulation. They are documented here so their contracts remain
visible to Documenter's completeness check; they are not public API.

```@docs
DRModels._LSWhitenedSeed
DRModels._ls_whitened_eval
DRModels._ls_whitened_information
DRModels._ls_whitened_vcov
DRModels._ls_bootstrap_effect
```

### Location-scale inner-mode acceptance

For the non-Gaussian location-scale Laplace engine, a positive-definite latent
Hessian alone does not certify a mode. The inner solver must also return finite
coordinates and a fresh finite gradient satisfying
`norm(gradient) <= tol * (1 + norm(a))`. The default remains `tol = 1e-9` with
at most 200 iterations. Failure to meet this condition is a failed inner solve,
not an accepted likelihood evaluation.

Near a mode, objective rounding can hide a useful Newton step. A separate local
polishing check permits a full, undamped step with an objective increase of at
most four units in the last place (ULPs), provided the actual displacement is small,
the gradient strictly decreases and meets the same stationarity criterion,
and the trial Hessian is finite and positive definite.

For NB2 and Gamma, a larger rounding discrepancy can trigger a second check
under the same safeguards. It estimates the objective change from directional
derivatives and a quadrature integral, avoiding subtraction of two nearly equal
objective values. The prior contribution retains multiplication and summation
residuals; its discarded rounding terms are tracked alongside the existing data
and quadrature error estimates. The estimated change plus its numerical error
margin must be negative. This fallback is unavailable at or across the kernels'
predictor-clamp boundaries, or when the tracked arithmetic risks overflow or
underflow. An unavailable or inconclusive estimate leaves ordinary backtracking
in place. The margin is an engineering estimate, not a proven error bound.

Neither polishing check guarantees exact objective descent or a global optimum.
Other steps retain the ordinary descent check; coordinate-identical trials do
not count as progress.

This helper is internal and has no stability guarantee.

```@docs
DRModels._ls_inner_estimated_change
```

### Phylogenetic group-index internals (no stability guarantee)

```@docs
DRModels._lss_phylo_group_index
```

## Non-Gaussian sparse-Laplace kernels

```@docs
DRModels._fit_beta_relmat_laplace
DRModels._fit_betabinomial_crossed_laplace
DRModels._fit_betabinomial_phylo_laplace
DRModels._fit_gamma_relmat_laplace
DRModels._fit_nb2_relmat_laplace
DRModels._fit_poisson_crossed_laplace
DRModels._fit_poisson_phylo_laplace
DRModels._fit_poisson_relmat_laplace
DRModels._fit_poisson_spatial_coord
DRModels._poisson_crossed_laplace_fg
DRModels._poisson_phylo_laplace_fg
```

## Helpers referenced by source docstrings

These nonpublic helpers have no source docstrings; the descriptions below keep
references from the documented kernels navigable without creating new APIs.

### [_group_index](@id _group_index)

Maps observation labels to integer group indices in first-seen order and
returns the indices together with the number of distinct groups.

### [_fit_phylo_mean_laplace_nuisance](@id _fit_phylo_mean_laplace_nuisance)

Prepares a tree's root-conditioned precision and observation-to-leaf map,
then delegates to the general-precision mean-effect Laplace fitter.

### [_fit_crossed_mean_laplace_nuisance](@id _fit_crossed_mean_laplace_nuisance)

Fits the crossed mean-effect Laplace objective for a supplied family and
nuisance-parameter specification, using the prepared group indices.


## Prepared missing-predictor development route

!!! warning "Experimental"
    Exported for evaluation, not yet stable. API and numerics may
    change; not covered by the R-parity scoreboard.

!!! warning "Limited developer interface"
    This prepared-array interface covers a Gaussian response with one Gaussian
    or Bernoulli predictor, or two independent Gaussian predictors. The
    likelihood has two **development** frontends: the [joint formula frontend](@ref joint-predictor-formula)
    and `drmTMB(..., engine = "julia")` through `drm_bridge_joint`. They admit
    a Gaussian identity-link response, one or two bare additive `mi()` terms, and
    complete fixed-effect exogenous designs. Grouped predictors, further
    predictor families, random or structured effects, REML, and all other
    missing-predictor models remain outside this route.

    The R preparation route supports `response = "drop"` or `"include"` with
    `predictor = "model"`; its response-drop preprocessing is deliberately not
    presented as native-TMB response-policy parity. Profile and bootstrap
    intervals are unavailable. Gaussian predictor-SD Wald intervals use a
    delta transformation on the natural SD scale and may cross zero; they are
    not a claim of native interval parity or interval coverage. Conditional
    variance at fixed parameters is not the native R `imputed()` standard
    error.

Design matrices must be complete. Only the modelled predictors and `y` may
contain `missing`. For the one-predictor interface, the parameter order is mean coefficients, the coefficient of `x`, residual
log-SD coefficients, predictor coefficients, and (Gaussian only) predictor
log-SD. This last coordinate is a log-SD, not R's natural-scale `sigma_mi_x`.

```@example prepared_joint
using DRModels
x = Union{Missing,Float64}[0.8, missing, 1.0, missing]
y = Union{Missing,Float64}[1.7, -0.2, missing, missing]
z = [-0.6, 0.3, 0.8, -0.1]
X = hcat(ones(4), z)
model = prepared_joint_model(y, x, X, ones(4, 1), X;
    predictor = :gaussian, mu_names = ["(Intercept)", "z"],
    predictor_names = ["(Intercept)", "z"], original_row = [17, 4, 81, 29])
theta = [0.2, -0.35, 0.7, 0.1, 0.0, 0.25, log(0.9)]
row_loglik = prepared_joint_rowloglik(model, theta)
moments = prepared_joint_conditional_moments(model, theta)
@assert row_loglik[4] == 0.0  # Both variables missing: integral equals one.
@assert moments.mean[1] == x[1]
(model.original_row, row_loglik, moments)
```

This example evaluates a supplied parameter vector; it does not fit its four
rows. On an identifiable dataset, `fit_prepared_joint(model)` estimates the
parameters. Inspect `joint_missing_summary(result).optimizer_status` and
`.covariance_status` separately. Its `.uncertainty_status` is
`:not_computed`: this low-level summary does not calculate standard errors.
Use [`imputed`](@ref) for native-shaped imputation summaries and their own
uncertainty status.

### Two independent Gaussian predictor models

The two-predictor array interface uses an `n × 2` predictor matrix and a tuple
of two complete predictor designs. It shares the prepared likelihood operations
above. Direct formula and R bridge admission use this same kernel; the bounded
route does not establish full native-R parity.

Independent predictor models can produce correlated conditional imputations:
when both predictors are missing, the observed response informs their joint
values. The returned covariance retains this dependence.

```@example prepared_joint_two
using DRModels
x = Union{Missing,Float64}[0.8 missing; missing missing; 1.0 0.3; missing missing]
y = Union{Missing,Float64}[1.7, -0.2, missing, missing]
z = [-0.6, 0.3, 0.8, -0.1]
X = hcat(ones(4), z)
model = prepared_joint_model(y, x, X, ones(4, 1), (X, X);
    predictor_variables = (:x1, :x2),
    mu_names = ["(Intercept)", "z"],
    predictor_names = (["(Intercept)", "z"], ["(Intercept)", "z"]))
# beta, b1, b2, residual log-SD, alpha1, logtau1, alpha2, logtau2
theta = [0.2, -0.35, 0.7, -0.4, 0.1, 0.0, 0.25, log(0.9), -0.2, 0.15, log(0.8)]
moments = prepared_joint_conditional_moments(model, theta)
@assert prepared_joint_rowloglik(model, theta)[4] == 0.0
@assert moments.covariance[2, 1, 2] > 0  # Opposite response slopes.
(moments.mean, moments.covariance[2, :, :])
```

This is a parameter-point calculation, not a fit of four observations.
On an identifiable dataset, `fit_prepared_joint(model)` fits the same prepared
model. Select a predictor explicitly with `imputed(result; variable = :x1)` or
`:x2`. Its Gaussian imputation SE combines conditional variance with first-order
parameter uncertainty using the full fitted covariance; it does not establish
interval coverage or provide multiple-imputation draws.

```@docs
DRModels.PreparedTwoJointGaussianModel
DRModels.PreparedTwoJointGaussianFit
DRModels.JointTwoMissingMetadata
```

### Ordinal and categorical predictor models

The finite-state prepared interface accepts one ordinal or categorical predictor
and a Gaussian response. It integrates over every possible state when the
predictor is missing. Supply an `n × K × p` array containing the complete mean
design for each row and state; ordinal contrasts and categorical dummy variables
must already be encoded in this array. The direct `mi()` formula frontend and R
bridge construct this state design for their bounded finite-state routes; they
do not establish full native-R prediction or accessor parity.

```@example prepared_joint_finite
using DRModels
levels = ["low", "middle", "high"]
x = Union{Missing,String}["middle", missing, "high", missing]
y = Union{Missing,Float64}[0.4, -0.3, missing, missing]
z = [-0.6, 0.3, 0.8, -0.1]
Xstate = zeros(4, 3, 2)
for i in 1:4, k in 1:3
    Xstate[i, k, :] = [1.0, k - 2.0]
end
model = prepared_joint_model(y, x, Xstate, ones(4, 1), reshape(z, 4, 1);
    predictor = :ordinal, levels = levels, variable = :severity)
# Mean coefficients, log-SD, predictor slope, first cut, log cut spacing.
theta = [0.1, 0.5, log(0.7), 0.3, -0.6, log(1.2)]
moments = prepared_joint_conditional_moments(model, theta)
@assert prepared_joint_rowloglik(model, theta)[4] == 0.0
@assert moments.mean[1] == 2.0
moments.probabilities
```

This example evaluates parameters, without fitting four rows. For ordinal
predictors, cumulative probabilities use `logistic(cutpoint - linear_predictor)`;
the predictor design must not span an intercept. Raw parameters are mean
coefficients, residual log-SD coefficients, predictor coefficients, then the first
cutpoint and log positive spacings. The raw covariance uses these same coordinates.

For `predictor = :categorical`, the first declared level is the baseline.
Predictor coefficients are ordered by nonbaseline level, then design term; there
is no cutpoint block. The prepared constructor accepts two or more states; that
low-level admission does not replace native frontend restrictions.

On identifiable datasets, `fit_prepared_joint(model)` returns state-weighted
fitted means and retains posterior probabilities. Ordinal `imputed()` reports
expected scores and conditional score SDs, without adding parameter uncertainty.
Categorical `imputed()` reports the first modal category code; a metric SE is
unavailable, with an explicit status. Fit covariance failures take precedence.
Neither summary is a multiple-imputation draw or an interval-coverage claim.

The direct formula frontend uses this same kernel. Supply declared levels for
textual ordered data; declaring nominal levels fixes the baseline explicitly:

```@example finite_formula
using DRModels, Random
rng = MersenneTwister(563)
labels = ["low", "medium", "high"]
codes = repeat(1:3, 30)
z = randn(rng, 90)
x = Union{Missing,String}[labels[k] for k in codes]
y = Union{Missing,Float64}[0.2 + 0.3*z[i] + 0.4*codes[i] + 0.5*randn(rng) for i in 1:90]
x[7:7:84] .= missing
y[14] = missing
data = (; y, x, z)
form = bf(@formula(y ~ z + mi(x)), @formula(sigma ~ 1))
predictor = impute_model(@formula(x ~ z);
    family = CumulativeLogit(), levels = ["low", "medium", "high"])
fit = drm(form, Gaussian(); data,
    impute = (x = predictor,),
    missing = miss_control(response = "include", predictor = "model"))
@assert isfinite(loglik(fit))
(cutpoints(fit), imputed(fit; rows = :missing))
```

Use `CategoricalLogit()` for a nominal predictor. In direct Julia, `coef(fit)`
and `vcov(fit)` retain the prepared **raw** parameter order, including ordinal
cutpoint coordinates; `cutpoints(fit)` returns constrained cutpoints separately.
This differs from R's public coefficient table, which omits predictor cutpoints.
The fitted likelihood is shared, but full accessor parity and the remaining
native default-fit discrepancies are still programme requirements.
No-intercept means follow R's first-factor coding: the first categorical main
effect receives full indicators, and later factors use reduced contrasts.
An ordinal missing predictor uses polynomial contrasts unless it is that first
factor. Complete numeric, plain-string, symbol and Boolean covariates are admitted.
Generated R state-expanded designs verify plain-string and Boolean factors
and their interactions, including coefficient names; symbol-valued factors
have not yet been separately checked against R.
Package-specific categorical/ordered value types require a typed contrast contract
and are refused when used in the mean formula. They remain required parity work.
Interactions containing `mi()` remain unsupported. R-prepared bridge designs
retain R's coding.

For new rows, `predict(fit, newdata)` uses the fitted design and its factor
contrasts. Supply a known predictor state; missing or unknown states are refused.
The prediction does not condition on a supplied new response. Training fitted
values remain available with `fitted(fit)`.

```@example finite_formula
newdata = (; z = [0.0, 0.5], x = ["low", "high"])
predict(fit, newdata)
# Residual SD needs only the columns used by the sigma formula.
predict(fit, (; z = [0.0, 0.5]); dpar = :sigma)
```

Use `type = :link` for the linear predictor or `type = :response` for the
response scale (the default). `se = true` returns predictions and delta-method
standard errors when the retained observed-information covariance is finite,
symmetric and positive definite. It refuses unavailable or invalid covariance;
point predictions with `se = false` remain available. This is not a prediction
interval, profile interval, or bootstrap. General missing-state new-data
integration and complete R accessor parity remain separate requirements.

```@docs
DRModels.PreparedFiniteJointModel
DRModels.PreparedFiniteJointFit
DRModels.JointFiniteMissingMetadata
DRModels._finite_joint_ordinal_logprobabilities
```

### Finite-state retained-prediction internals (no stability guarantee)

```@docs
DRModels._joint_finite_state_prediction_plan
```

```@docs
DRModels.PreparedJointModel
DRModels.PreparedJointFit
DRModels.prepared_joint_model
DRModels.prepared_joint_rowloglik
DRModels.prepared_joint_conditional_moments
DRModels.fit_prepared_joint
DRModels.joint_missing_summary
DRModels.JointMissingMetadata
DRModels.PreparedJointGaussian
DRModels.PreparedJointBernoulli
DRModels.prepared_joint_initial
DRModels._has_joint_mi
DRModels._fit_joint_formula
DRModels.drm_bridge_joint
```

### [prepared_joint_nll](@id prepared_joint_nll)

`prepared_joint_nll(model, theta)` returns the negative sum of row
log-likelihoods, retaining predictor-only observations and the exact zero
contribution of rows where both response and predictor are missing.
