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

Interval coverage is not claimed. There is no native comparison for
`profile_ci = true` under REML. It now profiles the restricted likelihood: the
other variance parameters are re-optimised, and beta is integrated out, not
profiled. Before this change it profiled the ML surface from the REML estimate.
`test_reml_sigma_phylo_joint.jl` checks that each finite endpoint sits at the
chi-square threshold on `_glsp_joint_reml_nll`.

**Variance boundary (review fix).** When a phylogenetic SD is estimated at
zero, the restricted NLL flattens onto a plateau. Its gradient falls like SD²,
below the rounding noise of the finite-difference gradient, so the outer Newton
crept toward the plateau until it hit the iteration cap and reported
non-convergence. On zero-signal sigma-only draws (20 tips x 4 rows, seeds
1001-1015), 6/15 fits converged with true SD 0, 7/15 with SD 0.1 and 12/15 with
SD 0.3; the base branch converged in 15/15. The fit is now declared converged
when three conditions hold: a log-SD is below -8, the gradient is below the
no-descent tolerance, and the last step gained at most 1e-10 relative. The
boundary coordinates are then moved onto the plateau's supremum. Now 15/15
converge at every SD. The reported logLik equals the SD -> 0 limit to within
1e-8, on both the sigma-only block and the coupled block with both SDs at zero.
The Wald covariance there is NaN, matching the ML routes' PD guard. None of the
fixtures in this receipt is on the boundary, and after the fix every row of
`julia.tsv` except the timings is byte-identical.

**Correlation bound (second review fix).** Native drmTMB bounds the phylo
correlation, `rho = 0.999999 * tanh(eta_cor_phylo)` (`drmTMB.cpp`). A reviewer
found two coupled fixtures whose native REML optimum sits on or at that bound,
where Julia stopped at a worse local optimum. `native-fit-boundary.R` writes
them (`fixture-G1/G2`, byte-identical to the reviewer's files) and
`native-boundary.tsv`. G1: seed 90210, 37 tips, 2-9 rows per species, n = 194,
unit height, `sigma ~ 1`. G2: seed 31337, 44 tips, n = 222, tree height 7.22,
`sigma ~ z`.

Why Julia missed it. Near the bound `P = Q x Lambda^-1` reaches 1e8 and beyond.
The inner mode then fails its absolute 1e-9 stationarity bound (the gradient's
rounding noise is ~eps*|P|*|a|), so `nll_R` was Inf from cor ~ 1 - 1e-5, and
where it was finite the prior quadratic cancelled to ~1e-8 of noise, too much
for the finite-difference Newton. The fix, on the coupled block only:
(1) the inner bound is raised to that noise floor, `max(1e-9, eps*|P|)`;
(2) the bound itself is fitted as a candidate, a 2-D Newton in
(log sd_mu, log sd_sigma) at cor = +/-0.999999, evaluated in whitened latent
coordinates (`a = L u`, `u ~ N(0, Q^-1 x I)`, loadings `Z L`); the Laplace
approximation is invariant to that change of variables, which a new test
checks against the direct form; (3) interior iterates past the bound do not
count. The bound wins only when its `nll_R` is lower by more than 1e-9
relative, and is then reported without a Wald covariance, like native.

| fixture | native REML logLik | julia before | julia now | df | coupled REML time before -> now |
|---|---|---|---|---|---|
| G1 | -333.85076879 | -334.82167833 (not converged; sd_mu 1.6e-5, cor -0.45) | -333.85076908 | 6 = 6 | 105 s -> 89 s |
| G2 | -812.27444353 | -812.27520364 (not converged; cor 0.9997) | -812.27444299 | 7 = 7 | 127 s -> 49 s |

The times include the coupled ML fit that seeds REML (75 s and 31 s); the
coupled ML route is unchanged. The fixed effects agree with native within
5e-7 relative. The SDs agree within 1e-4 relative once G2's are put on native's
unit-height scale (Julia reports them on the raw branch-length scale, times
sqrt(7.22), a pre-existing convention the fit warns about). G2's native
optimum is just inside the bound (cor 0.99999825); Julia's sits on it, 5.4e-7
higher in logLik.

The plateau rule was also made robust. The Julia 1.10 x86-64 CI shard failed
this file's zero-signal test (seed 1011 not converged), and seed 1019 failed
locally: the per-step gain hovered around the 1e-10 threshold. A fit is now
also declared converged on the boundary when a log-SD is below -6 (was -8), the
gradient is small, pushing the log-SD 6 units further out does not raise
`nll_R`, and the gradient in the other coordinates is below 1e-6 relative.
Zero-signal seeds 1001-1040 now all converge (1019 was the one failure);
interior fits are unchanged to 10 digits.

Guard: every row of `julia.tsv` (F1, F2) except the timings is byte-identical
after this fix, and on G1/G2 every ML, mean-only and sigma-only row is
identical to the pre-fix head. The coupled **ML** fits on G1/G2 still stop at
a worse optimum than native (G1 -331.94079 vs -330.92465); that is the
pre-existing coupled ML route, not changed here.

**Reproduce.**
```
DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold Rscript --no-init-file native-fit.R
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. docs/dev-log/evidence/arc2-gaussian-sigma-phylo-reml/julia-fit.jl
```
Run `native-fit.R` from this directory. It writes the fixtures and `native.tsv`.
`native-fit-boundary.R` writes the G fixtures and `native-boundary.tsv` the same way.
