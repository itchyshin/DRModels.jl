# Model specification

!!! note "Status — Reference"
    Mirrors drmTMB's [Model specification](https://itchyshin.github.io/drmTMB/reference/index.html) (13 items in drmTMB). A model is one [`bf`](@ref) formula bundle (one linear predictor per distributional parameter) plus a response family. All 13 drmTMB families are available.

## Formula bundle

```@docs
bf
DrmFormula
BivariateDrmFormula
```

## Response families

```@docs
Gaussian
Student
Poisson
NegBinomial2
TruncatedNegBinomial2
Beta
BetaBinomial
Binomial
Gamma
LogNormal
ZeroOneBeta
Tweedie
CumulativeLogit
```

## Advanced family type

`SkewNormal` is a Julia family type with its own docstring. Its presence here
does not widen the supported-model matrix; consult the capability matrix before
selecting a family and structure.

The source docstring below uses a shorthand formula sketch. For a call with
prepared data `dat`, use explicit Julia formula objects:

```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
          SkewNormal(); data = dat)
```

```@docs
SkewNormal
```

## Two-column response

```@docs
cbind
```

## [Modelled missing predictors](@id joint-predictor-formula)

!!! warning "Experimental"
    Exported for evaluation, not yet stable. API and numerics may
    change. This route is not available through the R bridge.

For a Gaussian response, `mi(x)` marks an additive predictor whose missing
values are integrated out under a joint model. Observed predictor values inform
the predictor distribution; missing predictors are not filled in before fitting.
This route supports one Gaussian or Bernoulli predictor, or two independent
Gaussian predictor models, with complete remaining exogenous fixed-effect
covariates. Neither the response nor either modelled predictor may appear in
those fixed designs. Other response families, three or more predictors, mixed
predictor families, random or structured effects, REML, and profile/bootstrap intervals
are not supported here. The same narrow model is available experimentally through
`drmTMB(..., engine = "julia")`; it does
not yet reproduce every fitted result and post-fit method from drmTMB's default
R/TMB engine.

```@example joint_formula
using DRModels, LinearAlgebra
BLAS.set_num_threads(1)
n = 32
z = collect(range(-1.2, 1.2; length=n))
x_full = 0.15 .+ 0.65 .* z .+ 0.15 .* sin.(1:n)
y_full = 0.3 .+ 0.4 .* z .+ 0.7 .* x_full .+ 0.18 .* cos.(1:n)
data = (y = Union{Missing,Float64}[i in (4, 24) ? missing : y_full[i] for i in 1:n],
        x = Union{Missing,Float64}[i in (8, 18, 24) ? missing : x_full[i] for i in 1:n], z=z)
fit = drm(bf(@formula(y ~ z + mi(x)), @formula(sigma ~ 1)), Gaussian();
    data=data, impute=(x=@formula(x ~ z),),
    missing=miss_control(response="include", predictor="model"))
@assert is_converged(fit)
@assert isposdef(Symmetric(vcov(fit)))
(coef(fit, :mu), coef(fit, :mi_x), coef(fit, :sigma_mi_x), imputed(fit))
```

For a binary predictor use
`impute=(x=impute_model(@formula(x ~ z); family=Binomial()),)`.
`miss_control(predictor="model")` retains the default `response="fail"` when
the response is complete; request `response="include"` for missing responses.
Unmarked incomplete predictors and unsupported options produce an error.

`imputed(fit; rows=:all)` retains original row numbers and includes observed
values, whose SEs are `missing`. Gaussian imputation SEs combine conditional
variance with a first-order parameter-uncertainty correction. Bernoulli summaries
return conditional probabilities and their Bernoulli SDs. These are neither
multiple-imputation draws nor interval-coverage guarantees. Check
`uncertainty_status`; `se=false` hides SEs without erasing a failure status.
`coef(fit, :sigma_mi_x)` returns natural predictor SD; unqualified `coef(fit)` and
`vcov(fit)` use the raw coordinates, including its log SD. Complete fixed-effect
interactions/transforms are allowed; interactions involving `mi(x)` are not yet
supported.

For the R bridge, write the corresponding R formula and use
`engine = "julia"`, `impute = list(x = x ~ z)` (or
`impute = list(x = impute_model(x ~ z, family = binomial()))` for a binary
predictor), and
`missing = miss_control(response = "drop", predictor = "model")` or
`miss_control(response = "include", predictor = "model")`.
Dropping responses before fitting behaves differently from drmTMB's default
R/TMB engine. Profile and bootstrap intervals are not available here. The
Gaussian predictor-SD Wald interval is a natural-scale delta interval, can cross
zero, and has not been shown to have reliable coverage across datasets or
sample sizes.

With two Gaussian predictors, mark each separately and provide both predictor
models. The `impute` entry order does not change which model belongs to which
variable. Their prior distributions are conditionally independent given their
covariates; conditioning on an observed response can correlate their missing
values. The prepared fit retains that full conditional covariance.

```@example joint_formula_two
using DRModels, Random, LinearAlgebra
BLAS.set_num_threads(1)
rng = MersenneTwister(9302)
n = 64
z = collect(range(-1, 1; length=n))
x1 = 0.2 .+ 0.4z .+ 0.5randn(rng, n)
x2 = -0.1 .+ 0.3z .+ 0.6randn(rng, n)
y = 0.3 .+ 0.6x1 .- 0.4x2 .+ 0.2z .+ 0.5randn(rng, n)
data = (y=Union{Missing,Float64}[i % 13 == 0 ? missing : y[i] for i in 1:n],
        x1=Union{Missing,Float64}[i % 5 == 0 ? missing : x1[i] for i in 1:n],
        x2=Union{Missing,Float64}[i % 7 == 0 ? missing : x2[i] for i in 1:n], z=z)
fit = drm(bf(@formula(y ~ z + mi(x1) + mi(x2))), Gaussian(); data=data,
    impute=(x2=@formula(x2 ~ z), x1=@formula(x1 ~ z)),
    missing=miss_control(response="include", predictor="model"))
@assert is_converged(fit)
(coef(fit, :sigma_mi_x1), coef(fit, :sigma_mi_x2),
 imputed(fit; variable=:x2))
```

With two predictors, `imputed` requires `variable`. Direct Julia returns raw
coefficients and covariance in the prepared order: exogenous mean coefficients,
the two marker slopes in formula order, residual scale coefficients, then each
predictor's coefficients and log SD. Natural-SD accessors do not change that raw
covariance convention. The R bridge instead restores native model-matrix column
order and transforms both predictor SDs and their full covariance to public
natural scales. In R use `impute=list(x1=x1~z, x2=x2~z)` and
`imputed(fit, variable="x2")`.

```@docs
mi
miss_control
impute_model
imputed
JointDrmFit
JointTwoDrmFit
JointMissingControl
JointImputeModel
```
