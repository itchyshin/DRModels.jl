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

**Speed (third review fix).** A reviewer timed coupled REML on a fixture with a
strong negative phylo correlation at about 29 minutes (1852 s and 1849 s, of
which the coupled ML seed was 83 s), against native's 0.68 s. `native-fit-speed.R`
writes that fixture (`fixture-H2`, byte-identical to the reviewer's files) and
`native-speed.tsv`. H2: seed 8675309, 41 tips, 1-12 rows per species with 3
singleton species, n = 299, tree height 3.58, true cor -0.95, `sigma ~ z`.

Measured, not inferred. The REML stage was run from the coupled ML estimate
with counters on every restricted-likelihood evaluation and every inner solve
(a scratch copy of the two REML functions; not part of the package). At the
previous head, the first interior start made 150 evaluations and 341 inner
solves in its first 60 s. 59 warm-started solves failed (17%) and took 47.0 s
of the 54.0 s spent evaluating (87%); the cold fallback then succeeded in about
10 ms, except 11 times. After that one evaluation ran for over 6 minutes inside
the fixed-effect line search, with every trial repeating the failing warm solve.

Why a warm solve failed. From the neighbouring mode, one Newton step took the
inner gradient from 2.4e-2 to 5.2e-7, and the next would take it to 6.7e-13.
That step raised the joint NLL by 4.4e-15 (5 ULP of 6.4), so the inner
solver's monotone line search rejected it, damped to its cap, repeated for 200
iterations (about 1.9 s) and reported failure.

The fix (`_glsp_joint_reml_nll`, REML only). From a warm start, plain Newton
steps come first. Their end point is accepted on the inner solver's own
certificate (gradient within the same tolerance, positive-definite Hessian),
and only if the gradient contracted at every step and the joint NLL ends no
higher than at the start beyond rounding. Anything else falls through to the
old warm-then-cold solve. The ML routes do not call this function.

After, same counters, H2 REML stage: interior starts 192 and 303 evaluations,
correlation-bound candidates 237, 61 and 119, final evaluation and Wald SEs 26.
6 of 1,146 interior inner solves missed the fast path and all were rescued by
the warm solve, with no cold solve.

| fixture | REML stage, cached ML start | end-to-end `drm(method = :REML)` | of which coupled ML seed | native REML |
|---|---|---|---|---|
| H2 | 2.7 s (was about 29 min) | 86.5 s | 83.0 s | 2.28 s |
| H2, unit height | 2.7 s (was about 9.5 min) | not rerun | 110.5 s | 2.36 s |
| G1 | 16.0 s | 89.4 s | 71.6 s | |
| G2 | | 40.6 s | 30.0 s | |
| F1 | | 13.1 s | 11.3 s | |

Local timings, Julia 1.13 aarch64, 2 threads, on a shared machine (native on
the same machine: 2.28 s for H2; the reviewer measured 0.68 s). REML-stage
times exclude compilation. The logLik, fixed effects, SDs and correlation are
unchanged on H2 (-52.64598679, cor -0.784410, as native), G1 and G2, and every
row of `julia.tsv` (F1, F2, all shapes and both estimators) is byte-identical
apart from its timing column. End to end, the coupled ML seed is now the cost.
That route is not changed here (DRModels issue #818).

Tests. The new "Arc 2 speed" testset runs the REML stage on H2 and G1 from the
recorded Julia ML estimates, checks native's logLik (1e-6), fixed effects
(1e-5), SDs (1e-4) and correlation, and fails if H2's stage takes over 300 s
(measured 24 s on the Julia 1.10 CI shard; G1's stage, 74 s there, has no bound).
The end-to-end cells whose coupled ML seed is slow (F1 coupled REML, the G1/G2
correlation-bound testset) run only with `DRM_SLOW_TESTS=1`. The file takes
64 s locally by default (191 s before) and 209 s with `DRM_SLOW_TESTS=1`. On
the Julia 1.10 CI shard 4/4 it took about 3 min 11 s at fcf6ca268 (about
31.5 min before), and the shard 18m16s (48m19s before).

**Separate block on H2 (fourth review fix).** With phylo() on both mu and
sigma, the default REML route (`phylo_coupled = false`, the separate block)
ended in an error on H2, "the joint mode failed at the optimum", at 4ff6cd6fa
(53 s and 40 s) and at e10cd752e (908 s). The outer search reached its optimum;
the error came from the final re-evaluation there, which re-solves the joint
mode from the cold start (the ML fixed effects with zero phylo effects). At
that optimum, v = [-0.97152563, -1.34433573], the cold inner mode fails at the
first step (`_ls_inner_mode` from a = 0 at the ML fixed effects returns
ok = false, also with 1,000 outer iterations), while every point 0.01 away
succeeds with a restricted NLL of about 54.694. Starting from the mode solved
at a neighbouring point succeeds at the optimum itself.

The fix (`_glsp_joint_reml_fit`). The fit keeps the joint mode (fixed effects
and phylo effects) that the winning search last solved. When the cold
re-evaluation at the optimum fails, it retries from that mode and errors only
if that also fails. Where the cold re-evaluation succeeds nothing changes, so
every row of `julia.tsv` is byte-identical apart from timings. H2 separate
REML now returns -54.6930595303, converged, in 53 s and 39 s end to end.

Where the time goes (a Julia profile of the end-to-end H2 separate REML fit).
About 90% of the samples are in the separate-block ML fit that seeds REML
(37 s on its own after compilation); the REML stage itself takes about 4 s.
Speeding up the separate-block REML end to end therefore needs a faster ML
seed. The ML route is not changed here.

Test. "Arc 2 separate block: H2 REML re-evaluates at its optimum" runs the
REML stage from the recorded Julia separate-block ML estimate (about 4 s),
checks convergence, the restricted NLL (54.6930595303, 1e-6) and a finite
Wald covariance, and checks that the cold solve at the optimum still fails, so
the retry is exercised. It fails without the fix. The end-to-end `drm()` cell
runs with `DRM_SLOW_TESTS=1` (58 s).

**Reproduce.**
```
DRMTMB_PATH=~/local-scratch/lanes/drmTMB-arc1-pr1304-fold Rscript --no-init-file native-fit.R
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. docs/dev-log/evidence/arc2-gaussian-sigma-phylo-reml/julia-fit.jl
```
Run `native-fit.R` from this directory. It writes the fixtures and `native.tsv`.
`native-fit-boundary.R` writes the G fixtures and `native-boundary.tsv` the same way;
`native-fit-speed.R` writes the H2 fixture and `native-speed.tsv`.
