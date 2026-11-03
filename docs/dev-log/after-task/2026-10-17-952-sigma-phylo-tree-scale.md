# After-task: Gaussian sigma-phylo route on tall trees (#952, lane n5)

## 1. Goal

`drm(bf(lsize ~ 1 + lrange + phylo(1 | species), sigma ~ 1 + phylo(1 | species)),
Gaussian(); tree = nwk)` on `ade4::carni70` (ultrametric, height 53.8) returned a fit
with `logLik = -1e18` and `converged = false`, at its start values. On the tree
rescaled to unit height it fitted, but 0.108 (0.1076) below drmTMB (the issue says
0.107).

Acceptance (Engine A):
1. red proof: no -1e18 returned as a fit; a finite logLik within 1e-3 of drmTMB
   -111.8411, or an error;
2. raw vs unit logLik equal to 1e-6;
3. siblings checked and listed;
4. full suite;
5. R/Julia parity against drmTMB `ca3ddef` on the same data;
6. the slow raw fit behind a slow test tag.

## 2. Implemented

`src/gaussian_locscale_phylo.jl` (the only engine file changed):

- `_glsp_tree_scale(Q)`: the tree height c = max diag(Q⁻¹) of the root-removed
  Brownian-motion precision, from `takahashi_diag` (O(p); 17 ms at p = 10,000 in
  the review's timing). It returns exactly 1 when the height is within 1e-6 of 1
  (the tolerance `_warn_if_tree_not_unit_height` uses) or when Q is not positive
  definite.
- `_fit_gaussian_locscale_phylo` is now a wrapper; the fitter body is renamed
  `_glsp_fit` and is otherwise unchanged. The wrapper fits on `c·Q` (every branch
  divided by c). This is exact: P = (cQ) ⊗ (cΛ_raw)⁻¹ = Q ⊗ Λ_raw⁻¹.
- `_glsp_unscale_fit` maps the fit back to the raw branch-length scale:
  - log SDs move by -½ log c (`:resd_mu`, `:resd_sigma`, and log L11 and log L22 of
    `:recov`);
  - L21 is multiplied by 1/√c;
  - the vcov takes the Jacobian D V D;
  - `lambda_sd_*` and `profile_ci_sd_*` are multiplied by 1/√c;
  - the user's penalty spec is restored on a MAP fit.
- `_glsp_rescale_penalty`: the PC prior on the raw-scale SD becomes rate/√c on the
  rescaled tree. This is an exact identity in value and log-SD derivative; the
  correlation term is scale free.

`src/boundary_diagnostics.jl`: the fit-failed warning prints `is_converged(fit)`
instead of the raw optimiser flag. The 15 Oct retest saw `converged = true` on a
sentinel fit.

## 3a. Decisions and rejected alternatives

- **Rescale inside the fitter, not only the start values.** Shifting the starts by
  -½ log c makes the start objective finite here (118.88 instead of the 1e18
  sentinel), and would likely fix this repro, though that was not run. But the
  coupled bound's SD floor (log 0.05), the REML boundary rule (-6), the REML
  starts and the Nelder-Mead simplex are absolute too. Rescaling Q makes all of
  them act on the unit-height scale, by construction. That is the scale most of
  this route's fixtures use, and the one drmTMB's `vcv(corr = TRUE)` gives an
  ultrametric tree. Consequence (intended): on a tree with height ≠ 1, a fit whose
  SD sits near one of those cuts can get a different vcov-NaN or boundary decision
  than on base, while θ is essentially unchanged.
- **Report on the raw scale, as before.** The alternative was to report
  correlation-scale SDs and set `phylo_scale = :correlation`. That would change the
  reporting contract and the bootstrap simulator, and it is wrong for
  non-ultrametric trees (drmTMB's `vcv(corr = TRUE)` is tip-wise, not one scalar).
  So on the raw tree the SDs, like a PC prior's `sd_u`, stay on the branch-length
  scale. Matching drmTMB's `sd_u = 1` on this tree needs `sd_u = 1/√53.8`.
- **The default stays the separate block.** The issue's gap to drmTMB is a model
  difference, not an early stop. The default fits no mu-sigma phylo correlation
  (5 parameters); drmTMB estimates one (df = 6). Making the coupled block the
  default would change the formula/likelihood contract (Boole/Hopper), so it is
  left to the owner.
- **Sentinel stays a warning, not an error.** #1054 added a fit-time warning for
  sentinel fits on the Gaussian `drm` entry, the only caller of
  `_warn_variance_boundary`. Other routes, such as the Gamma coupled route in §4,
  do not warn. This PR removes the cause on this route and leaves that contract
  as it is.
- **Not based on #1075.** That PR adds `_finite_hessian_from_gradient` and moves
  the crossed-mean SE path onto it; neither is touched here.
- **Test placement.** The raw-vs-unit and drmTMB cells are behind `DRM_SLOW_TESTS=1`,
  as the lane asked. One default cell, the issue's own raw-tree `drm()` call, is a
  deliberate deviation. CI never sets `DRM_SLOW_TESTS`, so without that cell a
  revert of the rescale would keep CI green (review, tests lens). It is a 4 s fit
  once compiled; the cell took 56.5 s in a fresh local shard, compilation included.

## 3b. Evidence

All runs used Julia 1.12.6 (conda-forge); R 4.5.3, TMB 1.9.23, and drmTMB `ca3ddef`
built from a clone outside the repo. Drivers and logs are in
`docs/dev-log/evidence/952-sigma-phylo-tree-scale/` (see its `receipt.md`).

Before the fix, on base `f71b292`, raw tree (H = 53.8):

| carni70 | raw tree | unit tree |
|---|---|---|
| separate (default) | -1e18, not converged, start values, 157.7 s | -111.94867803 |
| coupled | -119.53109852, not converged, 86.6 s | -111.84106326 |
| separate REML | throws "no start produced a finite restricted likelihood" | -113.7669083 |
| coupled REML | -123.3824011, not converged | -113.7298937 |
| separate MAP, coupled MAP, coupled MAP + `cor_sd`, separate `profile_ci` | -1e18 each | finite |

After the fix, raw vs unit:

| Quantity | Agreement |
|---|---|
| logLik, ML and REML (separate, coupled, sigma-only) | ≤ 1.9e-10 |
| logLik, MAP and profile | all 10 printed digits |
| β | 1.5e-11 (ML, MAP); 2.3e-9 (REML) |
| SE(β) | 6e-11 (ML) |
| raw SD × √53.8 vs unit SD | 3e-11 (ML); 5e-10 (MAP); 1.1e-8 (REML) |
| finite profile endpoint (sd_mu lower) | 2e-10 |

The profile upper endpoints are NaN (unresolved) on both trees. A raw ML fit takes
4.0 s (separate) or 5.5 s (coupled) once compiled; raw REML, MAP and profile fits
took 8 to 226 s, similar to the same fits on the unit tree.

Against drmTMB `ca3ddef` (its values are identical on the raw and unit trees):

| Model | DRModels.jl | drmTMB |
|---|---|---|
| coupled ML | -111.8410632557 | -111.84106326 (df 6) |
| separate ML | -111.9486780307 | -111.94867803, its objective with `eta_cor_phylo` held at 0 |
| coupled REML | -113.7298937 | -113.7298937 (df 6) |
| separate REML | -113.7669083 | -113.7669083, its REML objective with the correlation held at 0 |
| sigma-only ML / REML | match to 5.5e-9 / 1e-11 (REML -117.3401155) | -117.3401155 REML (df 4) |
| coupled MAP, `sd_u = 1` (unit tree; `sd_u = 1/√53.8` raw) | -111.9076795, penalty 2.592394557 | identical |
| coupled MAP + `cor_sd = 1` | -111.9164, penalty 3.517633443 | -111.9164, penalty 3.517633448 |
| separate MAP, against drmTMB's correlation-0 twin | -112.0100198, penalty 2.565808132 | -112.0100198, penalty 2.565808076 |

Notes on the table:
- The coupled β match drmTMB's `opt$par` to 4e-11 on both trees.
- The SDs match on the unit tree. On the raw tree DRModels.jl reports them on the
  branch-length scale (drmTMB's × 1/√53.8), as before.
- cor = 0.999999·tanh(η) = 0.238655.
- The two correlation-0 references are `nlminb` runs that returned code 1 at
  `rel.tol = 1e-14`, with max|grad| 1.2e-5 (ML) and 2.4e-6 (REML).

### Tests

- New `test/test_952_sigma_phylo_tree_scale.jl` on the fix: 39/39 default cells
  (local shard 3) and 66/66 with `DRM_SLOW_TESTS=1`
  (`evidence/.../core/test_slow_on_fix.log`).
- The committed file on base `f71b292`:
  - default cells: 2 passed, 4 failed, 8 errored. The 8 errors are the new helpers
    being undefined. The 4 failures are the warning flag and the three assertions
    on the issue's raw-tree call (-1e18).
  - with `DRM_SLOW_TESTS=1`: 14 passed, 19 failed, 8 errored: the raw-tree separate and coupled cells fail
    (-1e18; -119.53), and the sigma-only cell passes on base, because that block was
    already height-invariant.
- q=4 engine (`bench/run_sparse_tmb_nd.jl`): logLik -256.5064, converged. The q=4
  files are untouched. The wall time (2.76 s) came from a loaded machine with no
  base timing beside it, so speed was not compared.

### Full suite

- Local, all 372 files, run as the four CI shards (`DRM_TEST_SHARD=k/4`,
  `--check-bounds=yes`): 16,672 passed, 5 failed, 0 errors.
- Shards 3 and 4 stopped at their first failing file. The 110 files after those
  points were run by a per-file driver with the same preamble.
- The 5 failures are in 4 files, all outside this route. Each fails the same
  assertions on base `f71b292` in this environment, with identical printed values
  where any are printed, and each passes on CI:
  - `test_bridge_bootstrap_tree.jl:28-29` (non-Gaussian bridge SEs);
  - `test_bootstrap_formula_structured.jl:67` (Poisson phylo-mean bootstrap interval);
  - `test_q4_reml_vcov.jl:148` (relerr 0.001036 vs a 1e-3 bar);
  - `test_reml_q4_missing_response.jl:210`.
- CI on `78efb3e`: all 8 shards green. That head had no default cell exercising the
  fix. CI on `decad9b`, the first head with one, was still running when this was
  written.

### Review (multi-lens workflow; its skeptic-verification phase did not complete)

- **Math lens:** no defect.
- **Consumers lens:** no consumer made inconsistent. It noted the intended
  boundary-heuristics change in §3a, and three pre-existing behaviours unchanged by
  this PR:
  - the bootstrap refit does not forward `phylo_coupled` or `penalty`, so a coupled
    seed is refitted as the separate block;
  - sigma-phylo bootstrap draws are conditional, with no phylo field;
  - `ranef` on a separate-block fit returns an empty Dict.
- **Edges lens:** `_glsp_tree_scale` = `phylo_tree_height` on random non-ultrametric
  trees to 4e-14. One non-ultrametric case was a no-signal boundary fit (height
  33.2, all SDs about 1e-9). There the coupled raw and unit logLik differed by
  7e-8, and `is_converged` flipped between the two trees for the separate and
  coupled fits. Both trees give essentially the same problem (c·Q_raw = Q_unit up to
  rounding), so this is rounding on a flat surface, not the wrapper.
- **Tests lens:** 3 findings, fixed in `decad9b`: the default-suite guard, the data
  citation, and the test header wording.

## 4. Siblings (carni70, base vs fix, raw vs unit)

| Route | Tree-height dependent on base? | After this PR |
|---|---|---|
| Gaussian mu-only `phylo(1 \| sp)` (`:auto`, `:sparse`, `:sparse_lbfgs`, `:em`, `:gls`, REML) | no (≤ 7e-10); = drmTMB (ML 10 digits, REML 1e-7) | unchanged (different engine) |
| Gaussian sigma-only (asymmetric) block, ML / REML | no (2e-10) | numerically unchanged (logLik ≤ 1.4e-12, θ ≤ 3e-9), now via the rescaled tree |
| Gaussian both-phylo separate / coupled, ML / REML / MAP / profile | **yes** (above) | fixed |
| Student | no phylo route. `tree =` raises a MethodError, sigma or nu phylo is rejected, and the coupled `(1 \| p \| phylo(sp))` form raises an internal FieldError. drmTMB refuses mu and sigma phylo too, but fits `nu ~ phylo(1 \| sp)`. | unchanged |
| Gamma mu-only phylo, `sigma ~ 1` and `sigma ~ x` | no (7e-10); = drmTMB to 1.1e-8 | unchanged |
| **Gamma coupled `(1 \| p \| phylo(sp))` on mu and sigma** (`_fit_locscale`) | **yes.** Raw tree: -1e18 at the start values, not converged, no warning, about 100 s. Unit tree: -187.3208, itself at the cor = 1 boundary. drmTMB refuses this model. | **not fixed (follow-up)** |
| NB2 coupled `(1 \| p \| phylo(sp))` | static read, not measured: same `_fit_locscale` start on raw Q; likelihood bounded | not fixed |
| Bivariate q=4 phylo (and bivariate LogNormal through it) | static read, not measured: absolute Λ0 on raw Q, no height warning | not touched (verified engine) |
| Mean-only raw-Q routes (non-Gaussian sparse Laplace, cumulative, bivariate q=2, `meta_V`) | static read: low risk (no scale-axis latent) | not touched |

## 5. Acceptance status

1. Met for `phylo_coupled = true` (-111.8410632557 on both trees). The issue's
   default call no longer returns -1e18: it is finite and converged at -111.9487,
   which is drmTMB's model with the correlation held at 0, not drmTMB's native
   -111.8411. Owner decision below.
2. Met to ≤ 1.9e-10 for ML and REML. The test asserts it under `DRM_SLOW_TESTS=1`
   for ML only; the REML, MAP and profile invariance rests on the survey logs.
3. Done (§4).
4. Done, locally and on CI (§3b).
5. Done; evidence directory as above.
6. Deviated by one default cell (§3a), pending sign-off.

Worked example (DoD 4): not added as a docs page. This is an internal fix with no
API change, and the issue's call runs as the default test cell
`#952: the issue's default fit on the raw tree`.

## 6. Follow-ups

1. Owner decision: should the default for `phylo(1 | g)` on both mu and sigma be
   the coupled block, to match drmTMB's model for the same formula?
2. Rescale the tree in `_fit_locscale` (Gamma/NB2 coupled phylo). The survey showed
   a raw-tree fit from the rescaled start reaches the unit optimum within 2e-9. Its
   `LocScaleObjective` carries Q for profile and bootstrap, so the change is not a
   one-liner. Related: #996 (non-Gaussian phylo routes and tree height).
3. Measure the q=4 bivariate route on a tall tree.
4. The separate block's ML profile upper endpoints are NaN (unresolved) on carni70;
   drmTMB's correlation-0 twin gives 1.398 for sd_sigma. This predates this PR.
5. Bootstrap refits drop `phylo_coupled` and `penalty`. This predates this PR.
6. Student error parity: give `tree =` and the coupled form designed refusals.
