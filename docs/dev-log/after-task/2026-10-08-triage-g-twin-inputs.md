# 2026-10-08 — triage G twins (#1018, #1008, #962)

Shannon, with Noether (interval and bootstrap contract), Boole (no formula-grammar
change), Hopper (bridge status), and Rose (claim versus evidence). No spawned
subagents.

## What changed

- **#1018.** `_validate_ci_level` / `_validate_bootstrap_B` in `src/ci_args.jl`.
  Public interval entry points reject `level` outside `(0, 1)` and non-finite
  values, and bootstrap rejects `B < 1` or a non-integer `B`, before `quantile`
  can invert an interval or throw a bare `DomainError`.
- **#1008.** `update` merges the options `DrmFit` stores (`method = :REML`,
  non-default `marginal`, `penalty`) ahead of caller keywords. Unnamed positional
  extras error. Bootstrap refits forward a stored MAP penalty the same way.
- **#962.** Public bootstrap defaults `check_converged` to `true`. A dropped
  replicate sets `status = "bootstrap_incomplete"` and warns. Draws piled on a
  random-effect SD or correlation bound set `bootstrap_at_boundary` and still
  warn when any replicate was dropped. The q=4 bridge no longer passes
  `check_converged = false`.

## Behaviour change

Direct `bootstrap_ci` / `bootstrap_summary` / `bootstrap_result` calls that used
to keep unconverged replicates now drop them, set `bootstrap_incomplete`, and
warn. `failures = :error` (still the default) aborts only when a refit throws.
Gaussian location-scale, random-intercept, and location-scale-scale fits report
`converged` only when `Optim.converged` is true, the iteration limit was not
reached, and the Newton step in standard-error units is small
(`max |H⁻¹g| / SE ≤ 1e-3`). `Optim.g_converged` (absolute `1e-8`) is not part
of that flag. The absolute `|g| ≤ 1e-3` fallback remains when the Hessian is
not usable; the negative-eigenvalue branch in `optim_minimum_guard.jl` is
unchanged. A true group SD of 0 keeps 40 of 40 replicates and is
`bootstrap_at_boundary`. `lrt_boundary` now refuses different `nobs`, different
marginal approximations, and a penalized MAP fit, the same three guards
`lrtest` already had. Pass `check_converged = false` to recover the old
percentile.

A temporal Gaussian with its residual SD on the `sigma_ratio` boundary is
`is_converged` when the optimiser converged. That split is the documented
temporal boundary, not the saturated-mean collapse. Bootstrap of
`test/test_temporal_ar1.jl` was dropping all 8 replicates on that bar.

## Follow-ups filed, not in this commit

- #1046 — unpack `:recov` / `:phylocov` (blocks the `(1 + x | g)` slope-variance-0 test).
- #1047 — EM `iterations < 500` counts as converged (`src/location_only.jl:3391`).
- #1048 — `test/test_bootstrap.jl` phylo checks still pass `check_converged = false`.
- #1049 — absolute `|g| ≤ 1e-3` fallback is not unit-free; rescale `H` by its diagonal.
- #1050 — about 60 `drm_optim_converged` sites still use absolute `1e-8`.
- #1051 — `fit_mixed_family` bootstrap drops failed refits with no status.
- #1052 — a thrown refit still aborts (`failures = :error`).
- #1053 — the `nobs` guard compares counts only.

The minimum-success floor is untouched. `used == 0` still throws. Any floor above that is pending Shinichi.

## Not stored, so not carried by `update`

`K`, `A`, `tree`, `coords`, `algorithm`, `g_tol`, `profile_ci`,
`phylo_coupled`, `sparse`, `impute`, `missing`. The caller passes them again.
Cholesky blocks `:recov` and `:phylocov` are not classified by the bootstrap
boundary flag. `fit_mixed_family`'s `B = 0` remains "do not bootstrap".

## Verification

Local Julia 1.10.12, targeted files only (the full suite is the CI shard matrix):

- `test/test_triage_g_twin_inputs.jl` — 135 passed (55.4s), including `x * 1000` at 40/40 and group SD 0 at 40/40 with `bootstrap_at_boundary`
- `test/test_temporal_ar1.jl` — 130 passed (50.3s). CI run 37725449141 had failed this file on shard 4/4 (all 8 bootstrap replicates dropped). The LSS contract in that same run was 60/60 on both Julia versions.
- `test/test_comparison.jl` — 12 passed
- `test/test_bootstrap.jl` — 46 passed
- `test/test_ranef_varying_scale_convergence.jl` — 6 passed
- Earlier on the first commit: `test_bootstrap_thread_flags` 2, `test_bridge_biv_inference` 71

The phylogenetic solver-control checks in `test_bootstrap.jl` pass
`check_converged = false`. That EM fixture stops at `g_tol = 1e-4` with
`converged = false` on a non-degenerate fit, and the assertions are about
forwarding `algorithm` and `g_tol`, not about the new default.

## Rose

Claims in the NEWS fragment match the regression tests in
`test/test_triage_g_twin_inputs.jl` and the related files above. No drmTMB
source is vendored. The `check_converged` default change is labelled as a
behaviour change. `tools/parity_ledger.py` is a capability census, not a
source-hash pin of these files, so it was not regenerated.
