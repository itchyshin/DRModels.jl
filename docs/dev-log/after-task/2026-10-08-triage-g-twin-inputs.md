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
`converged` from the Newton step in standard-error units
(`max |H⁻¹g| / SE ≤ 1e-3`), with an absolute `|g| ≤ 1e-3` fallback when the
Hessian is not usable. An absolute gradient of `1e-8` marked the same stationary
fit as failed once the predictors were rescaled. Pass `check_converged = false`
to recover the old percentile.

## Follow-ups named, not in this commit

- `:recov` and `:phylocov` draws are still not unpacked into SDs and
  correlations for the boundary flag. A `(1 + x | g)` slope-variance-0 case
  is the regression that belongs with that unpack.
- `algorithm = :em` still treats `iterations < 500` as converged
  (`src/location_only.jl`). The phylogenetic solver-control checks in
  `test/test_bootstrap.jl` still pass `check_converged = false` for that reason.

## Not stored, so not carried by `update`

`K`, `A`, `tree`, `coords`, `algorithm`, `g_tol`, `profile_ci`,
`phylo_coupled`, `sparse`, `impute`, `missing`. The caller passes them again.
Cholesky blocks `:recov` and `:phylocov` are not classified by the bootstrap
boundary flag. `fit_mixed_family`'s `B = 0` remains "do not bootstrap".

## Verification

Local Julia 1.10.12, targeted files only (the full suite is the CI shard matrix):

- `test/test_triage_g_twin_inputs.jl` — 117 passed
- `test/test_lss_bootstrap_contract.jl` — 60 passed
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
