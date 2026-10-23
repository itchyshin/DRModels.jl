# Structured-effect markers

!!! note "Status — Reference"
    Mirrors drmTMB's [Structured-effect markers](https://itchyshin.github.io/drmTMB/reference/index.html) (6 in drmTMB). These markers wrap a random-effect term inside a [`bf`](@ref) formula to give it a known correlation structure (phylogeny, space, pedigree, an arbitrary relatedness matrix) or a known sampling-variance (meta-analysis).

## Correlation-structured random effects

```@docs
phylo
spatial
animal
relmat
```

## Temporal random effects (AR1 / OU)

Wave 1 (Gaussian mean, `sigma ~ 1`, ML). The Julia spelling is positional
because `@formula` cannot carry keyword arguments: drmTMB's
`temporal(1 | id, time = occ, structure = "ar1")` is
`temporal(1 | id, occ, ar1)` here (the R bridge accepts drmTMB's spelling).

`fitted` and `predict` are population-level (`Xβ̂`; drmTMB's `fitted()` is
conditional, the conditional temporal effects are `ranef(fit)[:id]`).
`simulate` and `bootstrap_ci` draw a fresh temporal chain per series (and a
fresh `(1 | id)` intercept), as drmTMB's default `simulate()`; drmTMB refuses
the temporal bootstrap, so `bootstrap_ci` is an extension. Wald standard
errors are reported for every coordinate; drmTMB exposes only AR1
mean-coefficient Wald intervals, and no interval calibration is claimed.

```@docs
temporal
temporal_parameters
```

## Known sampling variance (meta-analysis)

```@docs
meta_V
meta_vcov_bivariate
MetaVcovBivariate
```

## Location–scale–scale submodel markers

```@docs
sd
sd_phylo
```

## Advanced tree preparation helpers

These exported helpers prepare or inspect phylogenetic covariance inputs for
advanced workflows. They do not make every tree or structured-effect
combination a supported model; use the capability page to check the combinations
available for your response family.

```@docs
augmented_phy
random_balanced_tree
random_caterpillar_tree
phylo_tree_height
augmented_tree_precision
sigma_phy_dense
phylo_correlation
```
