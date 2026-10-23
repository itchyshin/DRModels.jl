# Evidence receipt: #952 Gaussian sigma-phylo route on tall trees (lane n5)

Generated outputs only. drmTMB was cloned at `ca3ddefa5ea11bbfd3b765615dfbb89305d2485a` and
built outside the repository; no drmTMB source is copied here, and the R scripts only
call its exported API and the fitted TMB object (`f$obj$fn`, `f$obj$gr`).

Environment: Linux x86_64, 4 cores. Julia 1.12.6 from conda-forge, with packages
resolved from the General registry by git. R 4.5.3 (conda-forge), TMB 1.9.23, ade4
1.7-24, ape 5.8-1. Base = main `f71b292`; fix = `2b3e8f3` (src identical at `decad9b`).

## core/: the issue's model on ade4::carni70, raw tree (height 53.8) and unit-height tree

| File | What it is |
|---|---|
| `drmtmb_native.R` -> `drmtmb_native.log` | Prepares the data exactly as in #952 and writes `carni70.csv`, `carni70.nwk` and `carni70_unit.nwk` (ape `write.tree`; height 1.00000000008). It then fits drmTMB natively on both trees: logLik -111.84106326, df 6, `opt$par` printed. |
| `drmtmb_cor0.R` -> `drmtmb_cor0.log` | The same TMB objective with `eta_cor_phylo` held at 0, minimised over the other 5 parameters by `nlminb` from two starts: -111.94867803 on both trees. `nlminb` returned code 1, not its success code 0 (`rel.tol = 1e-14` is below what it can reach), with max\|grad\| 1.17e-5. |
| `julia_base_repro.jl` -> `julia_base.log` | Base: separate and coupled, both trees. Raw separate -1e18 (start values, 157.7 s); raw coupled -119.53109852, not converged (86.6 s). |
| `julia_raw_vs_unit.jl` -> `julia_fix.log` | Fix: the same four fits, plus raw-vs-unit deltas. \|Δ logLik\| 1.79e-10 (separate) and 1.48e-10 (coupled); max\|Δβ\| 1.5e-11; SD ratio × √53.8 = 1 ± 3e-11; cor 0.238655 on both trees. |
| `test_default_on_base.log` | The committed test file's default cells on base: 2 passed, 4 failed, 8 errored. The 8 errors are the new helpers being undefined; the 4 failures are the warning flag and the 3 assertions on the issue's raw-tree `drm()` call (-1e18). |
| `test_slow_on_fix.log`, `test_slow_on_base.log` | The committed test file with `DRM_SLOW_TESTS=1` on the fix (66/66) and on base (14 passed, 19 failed, 8 errored; the sigma-only cell passes on base). |
| `bench_q4.log` | `bench/run_sparse_tmb_nd.jl` on the fix: logLik -256.5064, converged. Its wall time comes from a shared, loaded machine with no base timing beside it, so speed is not compared. |

To rerun: run `Rscript drmtmb_native.R` in an empty directory, then
`drmtmb_cor0.R` and `julia --project=<DRModels checkout> julia_raw_vs_unit.jl` from
that same directory (the Julia drivers read the CSV and Newick files from `pwd()`).

## survey/: sibling routes, base vs fix, raw vs unit, with drmTMB where it has the model

One directory per route, each holding the drivers that were run and their logs. The
drivers are session records, so their data paths point at the session scratchpad;
the data are those written by `core/drmtmb_native.R`.

| Route | Result |
|---|---|
| `mu_only_gaussian` | Height-invariant on base and fix (≤ 7e-10) for `:auto`, `:sparse`, `:sparse_lbfgs`, `:em`, `:gls`, `sparse = true` and REML. Equals drmTMB: ML -104.1175756 to 10 digits, REML to about 1e-7. |
| `sigma_only_gaussian` | Height-invariant already on base. The fix moves it by ≤ 1.4e-12 in logLik and ≤ 3e-9 in θ. Equals drmTMB to 5.5e-9 (ML) and 1e-11 (REML -117.3401155). |
| `reml_variants` | Base raw tree: separate REML throws; coupled REML -123.3824011, not converged. Fix: raw = unit to ≤ 1.7e-10. Coupled -113.7298937 = drmTMB `REML = TRUE`. Separate -113.7669083 = drmTMB's REML objective with the correlation held at 0 (nlminb code 1, max\|grad\| 2.4e-6). |
| `penalty_profile_variants` | Base raw tree: -1e18 for separate MAP, coupled MAP, coupled MAP + `cor_sd` and separate `profile_ci`. Fix: raw (`sd_u = 1/√53.8`) = unit (`sd_u = 1`) to all 10 printed digits. Coupled MAP -111.9076795 with penalty 2.592394557, as in drmTMB; `cor_sd = 1` gives 3.517633443 vs drmTMB 3.517633448. Profile: the one finite endpoint scales by 1/√53.8; the upper endpoints are NaN on both trees, where drmTMB's correlation-0 twin gives 1.398 for sd_sigma. This survey agent stopped before it reported; its logs are complete. |
| `student_phylo_sigma` | No phylo route in DRModels.jl. Passing `tree =` raises a MethodError, sigma or nu phylo is rejected, and the coupled `(1 \| p \| phylo(sp))` form raises an internal FieldError. drmTMB also refuses mu and sigma phylo, but fits `nu ~ phylo(1 \| species)`. |
| `gamma_phylo_sigma` | Mu-only phylo is height-invariant and equals drmTMB to 1.1e-8. **Coupled `(1 \| p \| phylo(sp))` on mu and sigma (`_fit_locscale`) is tree-height dependent on base and fix.** On the raw tree it returns -1e18 at the start values, not converged, with no warning, after about 100 s. The unit tree gives -187.3208, at the cor = 1 boundary. drmTMB refuses this model. |
| `static_scan_notes.txt` | A static read, not measured. NB2 coupled shares the `_fit_locscale` start on raw Q, but its likelihood is bounded. The bivariate q=4 phylo route, and bivariate LogNormal through it, uses an absolute Λ0 on raw Q and gives no height warning. The mean-only raw-Q routes are low risk. |

## suite/: full test suite on the fix

The 372 files ran as the four CI shards (`DRM_TEST_SHARD=k/4`; file lists in
`shard{1..4}_files.txt`), with `--check-bounds=yes`. A top-level failure aborts a
shard. Shards 3 and 4 each stopped at their first failing file, so the files after
it were run by `resilient_driver.jl`: same preamble, one testset per file, and the
run continues past a failure. The status files are listed below.

- Result: 16,672 passed, 5 failed, 0 errors.
- The 5 failures are in 4 files, all outside this route:
  - `test_bridge_bootstrap_tree.jl:28-29`
  - `test_bootstrap_formula_structured.jl:67`
  - `test_q4_reml_vcov.jl:148`
  - `test_reml_q4_missing_response.jl:210`
- Each fails the same assertions on base `f71b292` in this environment
  (`base_reruns_summary.txt`; printed values are identical where any are printed).
- CI on `78efb3e` was green on all 8 shards.

Status files: `shard_status.txt`, `shard3_tail_status.txt`, `shard4_tail_status.txt`,
`base_reruns_summary.txt`, `fix_rerun_summary.txt`.
