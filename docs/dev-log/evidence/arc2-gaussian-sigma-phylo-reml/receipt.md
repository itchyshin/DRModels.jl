# Arc 2 receipt: Gaussian location-scale REML with `phylo()` on `sigma`

**Question.** Does `drm(method = :REML)` fit the same restricted likelihood as native
drmTMB (`engine = "tmb"`, `REML = TRUE`) when `sigma` carries a `phylo(1 | sp)`
random intercept, alone (`sigma_only`) or together with one on `mu`
(`mu_sigma`, where native also estimates the mean-scale phylo correlation)?

**What native does.** `drm_apply_estimator_spec()` (drmTMB `R/drmTMB.R`) adds
`beta_mu` and, because `sigma` has a variance component, `beta_sigma` to TMB's
random vector. Printed by `native-fit.R` on every REML fit here:
`TMB random = beta_mu,beta_sigma,u_phylo`. The outer objective is therefore one
Laplace approximation over (phylo effects, beta_mu, beta_sigma) jointly, with
flat priors on the betas, including the `-(p/2) log(2 pi)` constant.

**What changed in DRModels.jl.** The sigma-phylo REML routes used a
Patterson-Thompson composite (`nll_ML(beta, v) + 0.5 logdet d2 nll_ML / d beta2`,
minimised over beta and v). That differs from native in where beta sits, in which
Hessian enters the log-determinant, and in the constant. The new
`_glsp_joint_reml_fit` (`src/gaussian_locscale_phylo.jl`) maximises native's
quantity over the variance parameters only, and reports the joint mode of beta
at the REML variance estimates. The coupled block (`phylo_coupled = true`, the
block native fits for `mu_sigma`) now accepts REML. Before this change it threw.

**Fixtures.** F1 is the Arc 1 probe fixture (`sweep_fixture("gaussian")`, seed
20260924, n = 90, 15 tips). F2 is a real phylogenetic location-scale draw with
seed 11: 50 tips x 6 rows, true SDs 0.6/0.5 and correlation -0.4. Both trees are
rescaled to unit height, which leaves native's fit unchanged. The F1 probe
values -143.375030 and -143.290781 are reproduced. Native pin: drmTMB `709efbeb0`
(`drmTMB-arc1-pr1304-fold`). DRModels base: `da8b3f871`.

**Result** (`compare.tsv`). SAME means the same df, |dlogLik| <= 1e-6 and every
estimate within 1e-5 relative (the correlation within 1e-5 absolute).

| fixture | shape | REML native logLik | REML julia logLik | df | verdict |
|---|---|---|---|---|---|
| F1 | sigma_only | -143.37503003 | -143.37503003 | 4 = 4 | SAME |
| F1 | mu_sigma | -143.29078125 | -143.29078125 | 6 = 6 | SAME |
| F2 | sigma_only | -505.20733395 | -505.20733395 | 4 = 4 | SAME |
| F2 | mu_sigma | -495.18229435 | -495.18229435 | 6 = 6 | SAME |

Before the change, from the pristine base (`julia-base.tsv`, `ARC2_TAG=base`):
F1 sigma_only was -146.1213 and F2 sigma_only was -507.9573. The coupled REML
rows errored.

The REML Wald SEs of the three fixed effects agree with native's `sdreport`
within 2.7e-6 relative on F2 and 1.1e-4 on F1. They are reported in the table,
but the verdict does not use them.

**Neighbour guard.** Base and branch give byte-identical `julia*.tsv` rows for
every ML row and for the mean-only `phylo()` REML/ML rows (`mu_only`, a
different route). F2 `mu_only` is SAME. F1 `mu_only` is `SAME_BOUNDARY_SD`: the
phylo SD sits at zero on both engines (native 2e-5, julia 1.5e-4), and the
logLik agrees to 3e-8.

**Not covered.** F1 `mu_sigma` **ML** is DIFFERENT: native reports
non-convergence at cor = -1.0000, and Julia stops at -0.9989 with a logLik
1.1e-4 lower. That is the pre-existing coupled ML route at a correlation
boundary, which this change does not touch.

The separate block (`phylo_coupled = false`) now uses the same joint-Laplace
REML. It has no native twin, because native always estimates the correlation.

Interval coverage is not claimed. `profile_ci = true` under REML still profiles
the ML surface, which is pre-existing behaviour.

**Reproduce.**
```
DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold Rscript --no-init-file native-fit.R
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. docs/dev-log/evidence/arc2-gaussian-sigma-phylo-reml/julia-fit.jl
```
Run `native-fit.R` from this directory. It writes the fixtures and `native.tsv`.
