```@raw html
---
layout: home

hero:
  name: "DRModels.jl"
  text: "What varies besides the mean?"
  tagline: "A formula-first Julia twin of drmTMB: the mean, the scale, and boundary probabilities of a response each take their own formula, with random effects and sparse phylogenetic structure."
  image:
    src: /drmodels-full-logo.png
    alt: "DRModels.jl hexagonal badge with four overlapping response curves"
  actions:
    - theme: brand
      text: Fit a location–scale model
      link: /getting-started
    - theme: alt
      text: What can I fit today?
      link: /model-guides/model-map
    - theme: alt
      text: Evidence & limits
      link: /capabilities

features:
  - title: "A formula per parameter"
    details: "bf() bundles one formula per distributional parameter your family admits, so the mean and the spread are modelled separately. The scale formula is on log σ, and the parameter is named sigma, never tau. Zero-inflation, hurdle, and zero/one-inflation parameters take formulas too."
  - title: "Correlation between two responses"
    details: "In the two-response bundle, rho12 takes its own formula, so residual correlation can itself depend on a covariate. It is a bivariate parameter and is not available in the univariate bundle."
  - title: "Read each coefficient on its own scale"
    details: "Every parameter is reported on the scale named by its link. The capability matrix says which families, structures, and sparse combinations are tested before you interpret a route."
  - title: "A twin, not a port"
    details: "MIT-licensed fresh code; drmTMB (GPL) stays the reference. The optional R bridge is experimental and lists its admitted engine = \"julia\" cells; the R-side glue lives in drmTMB, and the default R engine remains TMB."
---
```

## Choose your analysis

Start with the scientific question, then choose one complete route:

- **Does a response's average or residual variability change with predictors?**
  Begin with [Getting started](getting-started.md). It fits a Gaussian
  location–scale model and explains coefficients for the mean and residual
  standard deviation.
- **Are observations related through a phylogeny?** Begin with
  [Phylogenetic structured effects](tutorials/phylogenetic-models.md). It shows
  how to supply a tree, fit a phylogenetic random effect, and interpret its
  scale.
- **Do studies have known sampling variances in addition to between-study
  heterogeneity?** Begin with [Mean effects and residual
  heterogeneity](tutorials/meta-analysis.md). This route is for Gaussian
  meta-analysis with supplied, known sampling variances.

For another response type or a more specialised structure, use
[What can I fit today?](model-guides/model-map.md) after one of these routes.

## Start with the location–scale model

The first question is not which optimiser to use. It is **which feature of the
response might change?** A mean-only model asks whether the expected response
changes. A distributional model can also ask whether its spread, its shape, or
its boundary probabilities change. DRModels.jl keeps those questions separate by
giving each admitted parameter its own formula.

Given a table `dat` with numeric columns `y` and `x`, fit a location–scale
model as below. [Get started](getting-started.md) includes the complete data
setup and a runnable example.

```julia
using DRModels

# y varies in BOTH its mean and its spread with x:
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = dat)

coef(fit, :mu)      # mean coefficients
coef(fit, :sigma)   # coefficients on log σ — how the spread changes with x
summary(fit)        # readable coefficient table
```

The formula bundle `bf(...)` (alias `drm_formula(...)`) collects one formula per
available distributional parameter. See [Get started](getting-started.md) for a
worked, runnable example, or [What can I fit today?](model-guides/model-map.md)
for the model-space guide and its route-specific boundaries.

!!! warning "The scale is modelled on the log scale"
    For a Gaussian fit, `coef(fit, :sigma)` are coefficients on **log σ**, the
    residual standard deviation. A slope of γ = 0.2 means the residual SD
    multiplies by exp(0.2) ≈ 1.22 per one-unit increase in `x`, and the residual
    variance by exp(0.4) ≈ 1.49. Other families keep the log link on `sigma` but
    interpret it on their own scale: for Gamma it is a coefficient of variation,
    for Beta it maps to a precision. The parameter is named `sigma`, matching
    drmTMB, never `tau`.

## What the fit estimates

In the model above, DRModels.jl estimates one coefficient vector for **μ** (the mean)
and one for **log σ** (the residual standard deviation). A positive σ coefficient
means a multiplicative increase in residual spread, not an additive change in
the mean. Other families expose other admitted parameters, but the same rule
holds: interpret a coefficient on the scale named by that parameter's link.

## Evidence and limitations

DRModels.jl is a pre-release package. Use the [capability matrix](capabilities.md) to
choose a tested family–structure combination, and the
[diagnostics and validation guides](diagnostics-and-validation/testing-likelihoods.md)
to see how it was checked. The verified sparse phylogenetic engine, profile
likelihood, and bootstrap routes are useful evidence — not blanket promises for
every family, data set, or interval. In particular, an interval method being
implemented does not claim universal calibrated coverage.

## Relation to drmTMB

DRModels.jl is the Julia twin of [drmTMB](https://itchyshin.github.io/drmTMB/): it
adopts a closely related `bf()` grammar and vocabulary so R users do not have to
relearn the model class. It is independent, MIT-licensed Julia code — not a port
of drmTMB's GPL source. The optional [R ↔ Julia bridge](r-julia-bridge.md) is
experimental and lists the admitted `engine = "julia"` cells; `engine = "tmb"`
remains the default R route and does not require Julia. The
[Rosetta page](rosetta.md) compares the two syntaxes directly.

---

*MIT licensed. A sister package to [drmTMB](https://itchyshin.github.io/drmTMB/)
(GPL) and [GLLVModels.jl](https://itchyshin.github.io/GLLVModels.jl). DRModels.jl is fresh code —
never a port of drmTMB's GPL source.*
