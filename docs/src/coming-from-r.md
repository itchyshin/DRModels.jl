# Coming from R

You can use DRModels.jl in two different ways.

If you already work in R with drmTMB, keep using its default R/TMB engine unless
you have a specific reason to try Julia. It needs no Julia installation and is
the established route for the full drmTMB workflow.

If you want to work directly in Julia, begin with the
[first model tutorial](getting-started.md). The model ideas are the same:
choose a response family, write a formula for the average response, and add a
formula for variability only when that is part of your question.

## Moving a familiar model to Julia

The [R and Julia vocabulary](rosetta.md) page puts common drmTMB and DRModels.jl
syntax side by side. It is the best next step when you already have an R model
that you understand and want to translate it by hand.

## The optional R bridge

drmTMB also has an optional `engine = "julia"` route for a limited set of
models. It lets you keep writing an R formula while fitting the model with
DRModels.jl. This is not required for either package, and it does not make every
drmTMB model available in Julia. Follow drmTMB's
[Julia-engine setup guide](https://itchyshin.github.io/drmTMB/articles/julia-engine.html)
only when the documented example matches your question.

For the current supported model types, use
[What is tested](capabilities.md). For models involving spatial, temporal, or
other specialised R workflows, stay with the documented native R route unless a
Julia example explicitly says otherwise.

## drmTMB articles without a DRModels.jl page

These drmTMB "Start here" articles have no DRModels.jl counterpart yet. Read
them on the drmTMB site, and check any Julia-specific route against
[What is tested](capabilities.md):

- [Capability and limits](https://itchyshin.github.io/drmTMB/articles/capability-and-limits.html)
- [First-week intervals](https://itchyshin.github.io/drmTMB/articles/first-week-intervals.html)
- [Function map cheatsheet](https://itchyshin.github.io/drmTMB/articles/function-map-cheatsheet.html)
- [Missing data](https://itchyshin.github.io/drmTMB/articles/missing-data.html)
- [Comparing with other packages](https://itchyshin.github.io/drmTMB/articles/comparing-with-other-packages.html)
