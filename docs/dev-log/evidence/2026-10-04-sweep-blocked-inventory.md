# Sweep inventory, 2026-10-04

Head at inventory: `f57412b40` (`#967` had merged; `#1037` was still open).

`Optim.converged` is the OR of the x, f, and g criteria. `#944` is the default-tolerance plateau: `Optim.Options(g_tol = g_tol)` leaves the f and x tolerances at 0, so a flat step sets `f_converged` or `x_converged` while the gradient is still large. The reported flag on those routes now uses `drm_optim_converged` in `src/optim_minimum_guard.jl`.

## Already guarded, now the same helper

- `src/gaussian_ranef.jl:659`
- `src/gaussian_meta.jl:185`

## Left alone: deliberate `f_reltol` stops (`#946`)

These routes stop on an objective change because the gradient sits away from `g_tol` at a variance boundary. Requiring `g_converged` would change the verified engine.

- `src/fit_q4_sparse_tmb.jl`
- `src/fisherz_q4.jl`
- `src/reml_q2.jl`
- `src/reml_q4.jl`
- `src/coevolution_q.jl` (also owned by open pull request `#793`)
- `src/locscale_profile.jl`
- `src/location_only.jl` (`f_reltol = 1e-9` on the sparse LBFGS spine, including the `best_res` flag)
- `src/experimental/` (not wired into the module)

## Left alone: open pull requests own the file

- `#770`: `src/beta.jl`, `src/gamma.jl`, `src/negbinomial.jl`, `test/runtests.jl`, `NEWS.md`
- `#793`: `src/gaussian_bivariate.jl`, `src/coevolution_q.jl`
- `#809`: `src/takahashi_selinv.jl`, `test/runtests.jl`
- `#1037`: `NEWS.md`, `Project.toml`
- Gate, even after `#967` merged: `src/locscale_fit.jl`, `src/locscale_inner.jl`

`test/runtests.jl` discovers `test_*.jl` on its own, so the new tests are not registered by editing that file.

## Left alone: not the reported flag

Laplace outer loops (`src/student.jl`, `src/ordinary_laplace.jl`), profile retries (`src/reml_q4.jl`, `src/heritability.jl`, `src/inference.jl`), and the mixed-family profile stall test (`src/mixed_family.jl` lines 170 and 176). The mixed-family reported flag at the `converged =` field does use the helper.

## Plateau flags switched to `drm_optim_converged`

67 call sites, then the two mixed-family control lines were put back. Nelder-Mead in this Optim version stores its own success in `g_converged`, so a real Nelder-Mead fallback still reports converged. An LBFGS plateau on the same function does not.

`src/skewnormal.jl`, `src/betabinomial.jl`, `src/gaussian_ranef.jl`, `src/student.jl` (the `DrmFit` lines only), `src/gaussian_core.jl`, `src/joint_missing_finite.jl`, `src/binomial.jl`, `src/lognormal.jl`, `src/joint_missing_two_predictor.jl`, `src/joint_missing_predictor.jl`, `src/cumulative.jl`, `src/variational.jl`, `src/gaussian_meta.jl`, `src/zeroonebeta.jl`, `src/poisson.jl`, `src/tweedie.jl`, `src/temporal.jl`, `src/gaussian_structured.jl`, `src/gaussian_sparse_lss.jl`, `src/truncated_poisson.jl`, `src/mixed_family.jl` (reported flag only), `src/gaussian_lss.jl`, `src/bivariate_student.jl`, `src/phylo_interaction.jl`, `src/locscale_sigma.jl`, `src/locscale_corr.jl`, `src/gaussian_locscale_phylo.jl`.

## Hessian and bootstrap, not this inventory's edit

`#972` is `src/vcov_guard.jl` (`minimum(abs, ev)` misses a negative eigenvalue). `#956` bypasses are `src/poisson.jl` spatial coordinates, `src/phylo_interaction.jl`, and `src/gaussian_bivariate.jl`. The bivariate file stays with `#793`. `#1038` and `#1025` are `src/inference.jl` `bootstrap_result`.
