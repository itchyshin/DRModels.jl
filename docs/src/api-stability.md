# API stability

This page explains which parts of DRModels.jl you can write into a long-lived
analysis and which parts may still change as the package develops.

## What stays stable

From `v0.7.0`, the **Stable** interface keeps its names, meanings, and
conventions across the `0.7.x` line and beyond. This includes:

- the `bf()` formula grammar and the `drm()` fitting function;
- the fourteen response-family constructors;
- structured-effect markers such as `phylo()`, `spatial()`, `animal()`, and
  `relmat()`; and
- the documented coefficient, inference, prediction, and plotting functions.

The parameter conventions also stay the same: residual scale is `sigma` (never
`tau`), bivariate residual correlation is `rho12`, and known sampling variance
for meta-analysis is supplied with `meta_V()`.

A future breaking change to this interface would require a major, clearly
announced version change. It will not arrive silently in a routine `0.7.x`
update.

## What is still experimental

Experimental functions work for their documented uses, but their arguments,
return values, or supported model classes may change between releases. They
include:

- `r2_constant_sigma()`, while broader definitions of marginal and conditional
  R² for random-effect models are still being designed;
- the R bridge (`drm_bridge`, `drm_bridge_inference`, and `drm_listwise`);
- cross-family and staged-pair tools (`mf_*`, `associate_pairs`,
  `latent_normal`, `association`, `PairAssociation`, and
  `integration_diagnostics`);
- penalised phylogenetic fits, bivariate meta-analysis, and the variational
  approximation selected by `marginal = :VA`; and
- joint models for missing predictors, including `mi()` and the prepared-model
  helpers.

If you use one of these in an analysis that must be reproducible for several
years, record the DRModels.jl version and read its release notes before updating.

## Engine functions

Some exported names are low-level computational building blocks for advanced
scripts and benchmarks. Examples include `AugProblem`, `make_problem`,
`fit_q4_sparse_tmb`, `estep_mode`, tree utilities, and parameter packers. They
are not part of the stable analysis interface. Most readers should use `bf()`,
`drm()`, and the documented post-fit functions instead.

## What the promise covers

The stability promise covers function names, argument meanings, and documented
return conventions. It does not require every optimiser to follow the same
numerical path: improvements to tolerances or algorithms may move estimates
within the model's documented numerical accuracy.

It also does not turn the existence of a confidence-interval function into a
general interval-coverage claim. Coverage depends on the model, sample size,
parameter, and data-generating process. Read the limits on the relevant model
page and assess the fitted model for your own study.

## Current exclusions

- Bivariate Student-t models do not accept `phylo`, `relmat`, `animal`, or
  `spatial` structured effects. Bivariate LogNormal models do support the
  structured effects listed on their model page because they are fit as
  Gaussian models on `log(y)`.
- Joint missing-predictor functions remain Experimental even though they are
  available through the public module.

For model-by-model boundaries, use
[Detailed capabilities and limits](capabilities.md). For a runnable first model,
start with [Getting started](getting-started.md).
