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
to keep unconverged replicates now treat them as failures. `failures = :error`
(still the default) aborts; `failures = :skip` drops them and warns. Intervals
can change for models that previously admitted diverged refits. Pass
`check_converged = false` to recover the old percentile.

## Not stored, so not carried by `update`

`K`, `A`, `tree`, `coords`, `algorithm`, `g_tol`, `profile_ci`,
`phylo_coupled`, `sparse`, `impute`, `missing`. The caller passes them again.
Cholesky blocks `:recov` and `:phylocov` are not classified by the bootstrap
boundary flag. `fit_mixed_family`'s `B = 0` remains "do not bootstrap".

## Verification

Local Julia 1.10.12, targeted files only (the full suite is the CI shard matrix):

- `test/test_triage_g_twin_inputs.jl` — 99 passed
- `test/test_comparison.jl` — 12 passed
- `test/test_bootstrap.jl` — 46 passed
- `test/test_bootstrap_thread_flags.jl` — 2 passed
- `test/test_bridge_biv_inference.jl` — 71 passed

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
