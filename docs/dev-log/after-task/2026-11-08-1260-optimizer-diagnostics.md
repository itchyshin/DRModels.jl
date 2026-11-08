# After-task: honest optimizer diagnostics (#1260)

Personas: Pat (diagnostics/docs), Rose (claim-vs-evidence). No spawned subagents.

Closes #1260 (review PR only; no merge).

## 1. Goal

Stop `bridge_diagnostics` from labelling every nonnegative `niterations` count `Optim.LBFGS`, and document that `niterations` is the accepted final Optim stage. Do not change fitted values, likelihoods, numerical algorithms, defaults, or the public fit API.

## 2. Implemented

- `src/introspection.jl`: `_bridge_optimizer_label` uses `control_info.actual_optimizer` when recorded (`:lbfgs` / `:bfgs` / `:newton`) via a function, not a mutable `Dict`. Unrecorded reports stay `missing`. Unrecorded Beta Newton polish is `missing`. Counted BFGS-first Laplace fingerprints (`indep_slopes_laplace` Gamma/Student colon-resd; `gaussian_rs_sigma_ranef` recov + `*_logsigma`) stay `missing` and are **not** labelled LBFGS. Remaining unrecorded counted non-Beta fits (Gaussian FE, Student ordinary RI Laplace, …) stay `Optim.LBFGS` by known-LBFGS proof, not by a nonnegative count. `Fminbox(LBFGS)` is still reported as `Optim.LBFGS` (doc note only). Multi-start uses the kept `actual_optimizer`, not `last(optimizer_runs)`.
- `src/gaussian_core.jl`: `niterations` docstring states final-stage vs total-stage semantics, BFGS selection, Beta Newton polish, the two BFGS-first counted routes, `Fminbox(LBFGS)` counts, and that Gaussian mean `(1 | g)` does not attach a count.
- `test/test_1260_optimizer_diagnostics.jl`: synthetic recorded-method / B1-fingerprint / kept-≠-last cases plus small live sibling routes (default Gaussian LBFGS, qualified BFGS, Beta unknown, Poisson Laplace `-1`, Gaussian RI BFGS with `-1` count, live Gamma independent slopes, live Gaussian rs-sigma).
- News / check-log fragments (this report). No `NEWS.md` edit.

Not edited (protected overlap): `src/beta.jl`, `src/control.jl`, `src/gaussian_ranef.jl`, `src/bridge.jl`, `src/indep_slopes_laplace.jl`, `src/gaussian_rs_sigma_ranef.jl` (no algorithm instrumentation), PR #1261 / #1214 / #1106 branches.

## 3a. Decisions and rejected alternatives

Recording Newton polish on `control_info` would be a public-report behaviour change and would touch `src/beta.jl` (open PRs #1214 and #1106). The issue allows "Newton or explicitly unknown"; unknown was chosen. Always opening a control scope so default fits record a method would make `control_info` non-`nothing` without `control=` (API change); stopped. No guessed new `DrmFit` field.

Independent review at `0b803be4` showed the remaining unrecorded-count → LBFGS fallback falsely labelled the two BFGS-first counted routes. Rejected: instrument those fitters (would change recorded algorithm/public report). Rejected: leave them as LBFGS. Chosen: exclude the two source-confirmed fingerprints, then LBFGS only for remaining known-LBFGS unrecorded counted non-Beta fits. `Fminbox` stays labelled `Optim.LBFGS` (inner method) with a docstring note rather than a new public token.

## 4. Files touched

`src/introspection.jl`, `src/gaussian_core.jl` (niterations docstring only; disjoint from PR #1261 hunks at `_drm_gaussian_fit` / `predict`), `test/test_1260_optimizer_diagnostics.jl`, `news/1260-optimizer-diagnostics.md`, `docs/dev-log/check-log.d/2026-11-08-1260-optimizer-diagnostics.md`, this report.

## 5. Checks run

Preflight: `HEAD` / `origin/main` pin `ed9185e953c5377f1de3b370ffffb02dbfe8192d` (no drift). `which julia` empty; no `/usr/bin/julia` / `/usr/local/bin/julia`; no `Manifest.toml` in the checkout. `nproc=4`. Shell PID recorded in the agent transcript. **No runtime Julia tests were executed.** Source inspection is not a runtime pass. Intended exact-head command, not run:

```
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. -e 'include("test/test_1260_optimizer_diagnostics.jl")'
```

Sibling files intended if Julia is present: `test/test_twin_gap_569.jl`, `test/test_optimizer_selection.jl`, `test/test_niterations.jl`. Full `Pkg.test()` omitted (no Julia; out of budget without a driver). Parity / `DRM_PARITY_TESTS` skipped (unauthorized).

`git diff --check` on the slice.

## 6. Tests of the tests

Synthetic cases fail if the helper still infers LBFGS from a nonnegative count, if BFGS is unlabeled, if an `:unrecorded` report becomes LBFGS, if unrecorded Beta polish is labelled LBFGS, if Gamma/Student independent-slopes or Gaussian rs-sigma fingerprints are labelled LBFGS, or if a multi_start report uses the last run instead of `actual_optimizer`. Default LBFGS is proved by a Gaussian FE fingerprint **and** a Student ordinary RI Laplace fingerprint (no colon), not by "any count". Live cases fail if default Gaussian is not LBFGS, if `DRMControl(optimizer=:bfgs)` is not `Optim.BFGS`, if Poisson Laplace fabricates a count, if Gaussian RI BFGS is unlabeled because `niterations == -1`, or if live B1 drm() routes label LBFGS. They do not prove Newton polish fired on the live Beta draw, and they were **not run** in this environment.

## 7a. Issue ledger

#1260 open, unassigned, no comments, no closing PR at start of this slice. Team A ownership candidate. No running DRMdriver. PR #1261 (`claude/q2-final-20261107`, updated 2026-11-08T02:28:52Z, repo push 2026-11-08T03:15:23Z) protected; gaussian_core.jl hunks start at line 1055 (q2/q3/q4 SD dispatch / predict), disjoint from the niterations docstring at 509–565. Other open PRs inspected for file overlap; this branch does not rebase or edit them.

## 8. Consistency audit / Rose

Claim: diagnostics/docs only. Evidence: no fitter, control selector, or default `DRMControl().optimizer` change; B1 files not edited. Remaining honesty: live Beta still does not *record* Newton, so the label is unknown even when polish was not accepted. Gaussian `(1 | g)` count stays `-1` (pre-existing; documented, not wired). B1 unrecorded counts are `missing`, not a fabricated BFGS or LBFGS. `Fminbox(LBFGS)` is still the string `Optim.LBFGS`. Docs no longer claim "never inferred from count" while inferring from count on B1 routes.

License: no drmTMB GPL source.

## 9. Follow-up source/API observations (read-only; not in this diff)

Pin: `ed9185e953c5377f1de3b370ffffb02dbfe8192d`. No extra numerical fits.

**`fit.coefnames` contract.** `DrmFit.coefnames::Vector{Pair{Symbol,Vector{String}}}` (`src/gaussian_core.jl`). The inner constructor `convert`s to that type. Accessors walk pairs (`_block_coefnames`, `coef(fit, param)`, `re_sd`, `Dict(fit.coefnames)`). `DRMControl` already tells users that `fit.coefnames` groups labels by parameter block. `String.(fit.coefnames)` is an external harness error (`String` has no method for `Pair{Symbol,Vector{String}}`), not a package failure. Flattening, when a caller wants it, is `reduce(vcat, last.(fit.coefnames))` or `collect(String, v)` per block — as existing tests already do. No extra API/doc change beyond #1260.

**Gaussian random-intercept `exp(logSD)` / boundary warning.** `_fit_ranef_gaussian` stores `lσb` in `θ` and uses `σb² = exp(2lσb)` (`src/gaussian_ranef.jl`). `re_sd` returns `exp(fit.theta[resd])`. This route does **not** apply the sparse-Laplace `[-8, 3]` log-SD clamp (#1190 / `_twin_floor_fixed_coords` in `src/convergence_status.jl`). A restart fires when `exp(lσb) < 1e-3 * scale_ref`; it does not clamp the coordinate to `-8`. Covariance uses `_vcov_from_hessian_checked` with context `"Gaussian (1 | g) location-scale"`, which `@warn`s a **clear** singular/non-PD Hessian (`src/vcov_guard.jl`: "Hessian is numerically singular at the optimum" / "not positive definite"). Older-pin pilot seed 20263112 at `ba502d9d` with logSD ≈ −22.79 and that warning is therefore **not** a silent #1190 clamp-at-−8 reproduction. No current-main numerical refit was run for this follow-up.

## 10. Residuals

Runtime tests unrun in this environment (`JULIA_ABSENT`). Exact-head live B1 / Fminbox / Beta polish / kept≠last plus `test_niterations.jl` / `test_optimizer_selection.jl` / `test_twin_gap_569.jl` / full `Pkg.test()` remain **BLOCKED** until a legitimate Julia runtime. Independent review required; no self-approval. Review-only PR; do not merge from this agent. No second repro launch.

## 11. Pin / branch

- Pin: `ed9185e953c5377f1de3b370ffffb02dbfe8192d`
- Branch: `cursor/niterations-optimizer-label-4ce4`
