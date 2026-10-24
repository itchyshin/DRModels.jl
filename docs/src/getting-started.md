# Getting started

DRModels.jl fits *distributional* regression models: the mean and the residual
standard deviation (SD) of a response each get their own formula, so a
covariate can change both the average and the spread. In formulas and in
`coef(fit, …)` the mean **μ** is called `mu` and the residual SD **σ** is called
`sigma`. Some response families have further parameters that can also take a
formula; see [Choosing response families](families.md).

!!! note "What you will do"
    Fit a Gaussian model in which a predictor changes both an outcome's average
    and its residual spread. You will then read the estimated changes, check
    whether the optimiser converged, and see what an interval means. For the
    full set of currently supported models, see
    [What can I fit today?](model-guides/model-map.md).

This page takes you from a clean Julia session to a fitted model and shows how to
turn its output into a scientific statement. The worked data are simulated so
you can see the intended pattern clearly; a successful fit here is practice,
not evidence about a real biological system.

## Install

This is the direct-Julia route. If Julia is new to you, first install **Julia
1.10 or later** and open its Julia prompt (the REPL). If you would rather keep
working in R, you do not need this setup: use the optional
[drmTMB Julia-engine route](https://itchyshin.github.io/drmTMB/articles/julia-engine.html)
instead.

Install the current package directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/itchyshin/DRModels.jl")
using DRModels
```

If you are developing DRModels.jl from a local clone, replace the installation
line with `Pkg.develop(path = "/absolute/path/to/your/DRModels.jl")`.

Some pages in this manual also load `Distributions` (to simulate data) or
`CairoMakie` and `AlgebraOfGraphics` (to draw figures). Julia lets you load
only packages you have added yourself, so add these once if a page uses them:

```julia
Pkg.add(["Distributions", "CairoMakie", "AlgebraOfGraphics"])
```

The two verbs you will use the most are exported at the top level:

- `bf(...)` — bundle one formula per distributional parameter (alias
  `drm_formula`). In glmmTMB terms, `bf(@formula(y ~ x), @formula(sigma ~ x))`
  is `glmmTMB(y ~ x, dispformula = ~ x)`; drmTMB and brms use the same `bf`
  name.
- `drm(formula, family; data = ...)` — fit the model by maximum likelihood.

## Fit a first model where variability matters

This is the same example as the drmTMB package's first model, on the same
data. 120 individuals were sampled in two habitats, forest and grassland.
Habitat and temperature shift mean growth, and habitat also changes how much
growth varies between individuals. A mean-only model would miss that second
result.

The data are simulated, with these true values: mean growth
`1 + 0.6·grassland + 0.4·temperature`, and log residual SD
`-0.5 + 0.45·grassland`. Julia's random numbers cannot reproduce R's
`set.seed(1)`, so the table is shipped with the package as
`test/data/habitat_growth.csv`, generated from drmTMB's code (provenance in
`test/data/README.md`). Reading it needs only base Julia:

```@example getstarted
using DRModels

path = pkgdir(DRModels, "test", "data", "habitat_growth.csv")
rows = split.(readlines(path)[2:end], ',')          # skip the header line
habitat     = String[r[1] for r in rows]
temperature = parse.(Float64, [r[2] for r in rows])
growth      = parse.(Float64, [r[3] for r in rows])
dat = (; growth, habitat, temperature)              # a named table of columns

fit = drm(bf(@formula(growth ~ habitat + temperature), @formula(sigma ~ habitat)),
          Gaussian(); data = dat)
```

For your own data, a `DataFrame` (for example read from a CSV file with CSV.jl)
works directly: `data = df`. Two pieces of Julia syntax appear above. `(; growth,
habitat, temperature)` makes a named table from existing vectors, and
`@formula(…)` writes a model formula, as `~` does in R.

`bf(@formula(growth ~ habitat + temperature), @formula(sigma ~ habitat))` is the
whole idea in one line: the first formula sets the response and the **μ**
sub-model; the second sets the **σ** sub-model. (Drop the `sigma` formula and it
defaults to `sigma ~ 1`, a constant residual SD — the ordinary regression that
`lm` or `lmer` fits.)

## Read the coefficients

`coef(fit, :param)` returns the coefficient block for one distributional
parameter. The mean block is on the **response scale** (intercept, grassland
contrast, temperature slope). The true values are `(1.0, 0.6, 0.4)`; the
estimates land within about one standard error of them:

```@example getstarted
coef(fit, :mu)
```

The scale block acts on **log σ** (intercept, grassland contrast). The true
values are `(-0.5, 0.45)`:

```@example getstarted
coef(fit, :sigma)
```

A positive grassland coefficient in the σ block means residual growth is more
variable in grassland than in forest. Because the σ link is logarithmic,
exponentiating a coefficient turns it into a **multiplicative effect on the
residual SD**:

```@example getstarted
exp(coef(fit, :sigma)[2])      # grassland / forest residual-SD ratio (true value 1.57)
```

Calling `coef(fit)` with no parameter concatenates every block into one vector
(μ first, then σ) — the raw parameter vector the optimiser returns.

## Read the model fit

`loglik(fit)` is the maximised log-likelihood; `aic` / `bic` build on it for
model comparison, and `nobs` / `dof` report the sample size and number of
estimated parameters:

```@example getstarted
(loglik = loglik(fit), aic = aic(fit), bic = bic(fit), nobs = nobs(fit), dof = dof(fit))
```

Always check that the optimiser actually converged before trusting any of the
above:

```@example getstarted
is_converged(fit)
```

`true` is necessary but not sufficient. It says the optimiser stopped by its
own rules; it does not prove the estimates sit at a well-defined maximum. A fit
can stop on a flat stretch of the likelihood, or with a parameter at its limit
(a variance at zero, a Student-t `nu` running to infinity), and still report
`true`. Its stricter checks for a collapsed residual SD apply to Gaussian fits
only. [`check_drm`](@ref) adds the size of the gradient (`max_abs_grad`) and
whether the covariance is usable; read its `grad_source` before trusting its
`ok` summary, and see [Improving convergence](model-guides/convergence.md).

`summary(fit)` prints estimates, standard errors, Wald intervals, and its
coefficient-level `z` and `p` summaries for every block at once. Treat those
summaries as evidence conditional on this fitted model, not as proof that a
biological effect is real; check diagnostics and the size of the estimated
scale change as well. Row names are prefixed with the parameter (`mu: …`,
`sigma: …`) so they stay unique:

```@example getstarted
summary(fit)
show(stdout, MIME"text/plain"(), summary(fit)); nothing # hide
```

## Confidence intervals

`confint(fit)` returns one row `(param, coef, estimate, lower, upper)` per
coefficient, on each block's working scale (μ on the response scale, σ on
`log σ`). The default is a 95% Wald interval:

```@example getstarted
confint(fit)
```

For a coefficient near a boundary, or whenever you want intervals that do not
assume a quadratic log-likelihood, ask for the **profile-likelihood** interval
instead. It re-optimises the other parameters at each fixed value and uses a
likelihood-ratio calibration. That calibration is asymptotic, so a profile
interval does not have universal exact coverage:

```@example getstarted
confint(fit; method = :profile)
```

## Beyond a first fit

The same front end also provides these next steps; the
[capability map](model-guides/model-map.md) records their current boundaries:

- **Per-parameter prediction** — `predict_parameters` (fitted μ/σ/… on new
  data), `marginal_parameters` (the stored in-sample values), and
  `prediction_grid` for building a swept `newdata` grid from a reference table.
  Predictions set any random effects to zero: they describe an average group,
  not an average over groups.
- **Auditable profile-likelihood CIs** — `profile_result` returns the full
  profile object behind `confint(fit; method = :profile)`.
- **Post-fit accessors** — `summary`, `family`, `is_converged`, `deviance`,
  `dof_residual`, and `rho12` (bivariate residual correlation).
- **Non-Gaussian phylogenetic random effects** — `phylo(1 | species, tree)` on
  the mean for Poisson, NegBinomial2, Gamma, Beta, and Binomial families
  (constant `sigma`), fitted with a Laplace approximation (the method glmmTMB
  also uses to integrate out random effects), plus crossed intercepts
  `(1 | g) + (1 | h)` for the same families.

## Random effects and the default approximation

A model with random effects, such as `y ~ x + (1 | g)`, has a likelihood that
integrates over them, and that integral has to be approximated. For an ordinary
random intercept on the mean of most non-Gaussian families, DRModels.jl's
default (`marginal = :LA`) uses adaptive Gauss–Hermite quadrature with 5 nodes
per group, whereas drmTMB uses the Laplace approximation. The two packages
therefore report different log-likelihoods for the same model, by about 0.001
to 0.24 units on the count, Beta, Gamma and Student datasets compared so far.
To reproduce drmTMB's numbers for Poisson, Binomial or Student, or for
NegBinomial2, Gamma or Beta with `sigma ~ 1`, pass `marginal = :Laplace`. That
option is for parity, not a better default: the adaptive quadrature is usually
the more accurate integral. Compare `loglik`,
`lrtest`, `aic` and `bic` only between fits that use the same approximation.
The [drmTMB parity page](drmtmb-parity.md#Default-marginal-approximation)
lists the defaults for the main random-effect routes and the models that
accept `:Laplace`.

## Where to go next

- [Coming from R](coming-from-r.md) — choosing native R, direct Julia, or the
  optional `engine = "julia"` route.
- [Rosetta (R ↔ Julia)](rosetta.md) — vocabulary and workflow translation.
- [Choosing response families](families.md) — the full list of response
  families and how to fit each one.
- [What can I fit today?](model-guides/model-map.md) — the live capability map.
- [When variance carries signal](tutorials/location-scale.md) — a deeper
  location–scale walkthrough.
