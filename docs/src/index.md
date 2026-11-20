```@raw html
---
layout: home

hero:
  name: "DRModels.jl"
  text: "What varies besides the mean?"
  tagline: "Distributional regression models ask how predictors change a response's average, variability, or probability of zero. DRModels.jl fits them directly in Julia; no R installation is needed."
  image:
    src: /drmodels-full-logo.png
    alt: "DRModels.jl hexagonal badge with four overlapping response curves"
  actions:
    - theme: brand
      text: Fit your first model
      link: /getting-started
    - theme: alt
      text: Choose a model
      link: /model-guides/model-map
    - theme: alt
      text: Supported models & limits
      link: /capabilities

features:
  - title: "Study averages and variability"
    details: "Does a treatment change the average outcome, how much individuals differ, or both? Give the average and the spread separate models."
  - title: "Account for related observations"
    details: "Measurements from the same group or related species need not be independent. Include group effects or relationships supplied by a phylogenetic tree."
  - title: "Combine evidence across studies"
    details: "Estimate an average effect across studies and examine why effects differ, while accounting for their known sampling variances."
---
```

## What is distributional regression?

An ordinary regression often asks how the **average response** changes with a
predictor. **Distributional regression** also asks whether other features of
the response change, such as its spread or probability of being zero. The
letters **DR** in DRModels refer to distributional regression.

For example, warmer conditions might change both average body size and how much
individuals differ in size. In psychology, a treatment might change both average
reaction time and its variability. These are separate questions, and a change in
one does not imply a change in the other.

DRModels.jl is a standalone Julia package for fitting these models to one or two
responses. It is general-purpose; the tutorials draw mainly on ecology,
evolution, environmental science, and evidence synthesis.

!!! warning "Experimental software"
    Check the [supported models and current limits](capabilities.md) before
    choosing a model, and check the fitted model before interpreting it.
    Validation is specific to the model and method used.

## Choose your analysis

Start with the scientific question, then choose one complete route:

- **Does a response's average or variability change with predictors?**
  Begin with [Getting started](getting-started.md). It fits a model for a
  continuous response, with separate formulas for the average and the variation
  left after accounting for that average.
- **Are observations related through a phylogeny?** Begin with
  [Phylogenetic structured effects](tutorials/phylogenetic-models.md). It shows
  how to supply a tree, account for shared ancestry, and interpret the results.
- **What is the average effect across studies, and why do effects differ?**
  Begin with [Mean effects and residual heterogeneity](tutorials/meta-analysis.md).
  This example combines study estimates with their known sampling variances.

For another response type or a more specialised structure, use
[What can I fit today?](model-guides/model-map.md) after one of these examples.

## Try a model of average and variability

This example matches the README and drmTMB's
[first-model article](https://itchyshin.github.io/drmTMB/articles/drmTMB.html):
growth of 120 individuals sampled in forest and grassland, where habitat and
temperature shift mean growth and habitat also changes residual variation.
These are simulated data, so the example illustrates the method rather than a
biological finding. The printed estimates will not match the R article, because
Julia and R generate different random numbers even from the same seed.

After [installing DRModels.jl](getting-started.md#Install), run:

```julia
using DRModels, Random
Random.seed!(1)

n = 120
habitat = repeat(["forest", "grassland"]; inner = n ÷ 2)
temperature = randn(n)
grass = habitat .== "grassland"
# true values: mean = 1 + 0.6*grassland + 0.4*temperature
#              log(sigma) = -0.5 + 0.45*grassland
growth = (1 .+ 0.6 .* grass .+ 0.4 .* temperature) .+
         exp.(-0.5 .+ 0.45 .* grass) .* randn(n)

fit = drm(bf(@formula(growth ~ 1 + habitat + temperature),   # mean (mu)
             @formula(sigma ~ 1 + habitat)),                 # log residual SD
          Gaussian(); data = (; growth, habitat, temperature))

coef(fit, :mu)      # true values: [1.00, 0.60, 0.40]
coef(fit, :sigma)   # true values: [-0.50, 0.45]
coeftable(fit)      # Wald SEs, z, p, 95% CIs for every coefficient
```

The first formula describes the average response. The second describes its
remaining spread, measured here by the log of the standard deviation (`sigma`).
The `sigma` habitat coefficient is a log residual-SD contrast: its true value
`0.45` means residual SD in grassland is `exp(0.45)`, about 1.57 times that in
forest. Fitted values will differ because the data include random variation.

The [first-model tutorial](getting-started.md) explains the formulas, how to
read estimates and confidence intervals, and what to check before reporting
results. This type of model is often called a **location–scale model**:
location describes the average and scale describes the spread.

## Evidence and limitations

Use [supported models and current limits](capabilities.md) to check which
combinations have been tested. After fitting, use
[Checking and using fitted models](model-guides/model-workflow.md) to decide
what to inspect and report. A method being available does not guarantee that
it will work well for every data set. In particular, the accuracy of confidence
intervals depends on the model, the data, and the method used to calculate them.

## Coming from R?

The R package [drmTMB](https://itchyshin.github.io/drmTMB/) also fits
distributional regression models. DRModels.jl uses related formula conventions,
but you can install and use it entirely within Julia. The
[Rosetta page](rosetta.md) compares the two syntaxes directly.

For R users who want to call Julia from R, the optional
[Coming from R](coming-from-r.md) explains the optional
`engine = "julia"` in drmTMB. That bridge is experimental; the two packages do
not support every model in the same way.

---

DRModels.jl is independently written Julia software, available under the MIT
license. For models of many responses together, see
[GLLVModels.jl](https://itchyshin.github.io/GLLVModels.jl).
