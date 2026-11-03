# After-task: #965 — `marginal = :VA` inner stationarity (lane n4)

## 1. Goal

#965: the five `marginal = :VA` random-intercept fitters (Poisson, Binomial,
NegBinomial2, Gamma, Beta `(1 | g)`) profile each group's variational mean `m`
and log-variance `r = log s` with an inner Newton solve. On base f71b292 that
solve was a fixed 25- (Poisson) or 30-step (GHQ families) unroll with a
coordinate-wise step clamp and no stationarity test, so the outer `converged`
flag could not see an inner point far from its optimum. Shannon, with Noether
(inner solver, kernels, convergence flag), Karpinski (cost of the extra inner
work, ForwardDiff through the unroll), Curie (repro search and the identity /
crash-rate harnesses) and Rose (claims against evidence, licence, parity note).
Scouting, adversarial review and the Rose audit ran as workflow subagents; none
run now.

## 2. Implemented

All in `src/variational.jl`.

- **One inner driver, `_va_inner_solve`**, used by all five families. Its first
  steps are the fixed `iters` unroll (one fewer when that last step would not move
  the iterate). It then continues, up to `maxiters = 200` steps in all, until the
  squared Newton decrement `λ² = gᵀ(−H)⁻¹g` is at most `1e-14`, takes up to two
  polishing steps after that, and stops early if the line search can no longer
  move the iterate. ForwardDiff differentiates through every step taken. It
  returns `ρ = λ² / max(1e-14, 32·eps·(1 + |F|))`, the decrement over the
  rounding floor of the group ELBO.
- **Step rules.** `_va_damped_step` damps only a Hessian that is not usably
  negative definite (`_va_negdef`: `a < 0`, `d < 0`, `det > 4·eps·(|ad| + b²)`).
  The old test, `det > eps·(1 + |Hmm| + |Hrr| + |Hmr|)`, checked only det, so it
  let a positive-definite `H` through (Beta); and it compared det with a sum of
  entries, so it damped a negative-definite `H` with a tiny `Hrr` (Poisson, huge
  `Σe^{η0}`). `_va_cap_step` caps the step along its own direction (`|Δm| ≤ 4`,
  `|Δr| ≤ 2`) instead of clamping each coordinate, which could turn the ascent
  direction into a descent one. Within `λ² ≤ 1e-8` the driver takes full Newton
  steps without the line search: that close, Newton converges quadratically, and
  the ELBO gain can fall below the rounding of F, where `_va_backtrack` cannot
  resolve it. `_va_backtrack` takes `F0` to avoid one evaluation per step.
- **Stop tests on primal values.** ForwardDiff 1.x `Dual ==` compares partials,
  so the stuck test and the decrement thresholds use `_va_primal`, and Float64
  and Dual evaluations take the same steps.
- **Kernels without cancellation.** Binomial `s·η − n·softplus(η)` is evaluated
  per sign of `η` with `log1p`. NB2's `b`-dependent part is
  `−y·log1p(r/μ) − r·log1p(μ/r)` (non-positive terms), and its `b`-free part
  `Σ loggamma(y + r) − loggamma(r)` is built exactly for `y ≤ 1000` and through
  the Stirling remainder above. Gamma uses `α·(expm1(−u) + u)`, with the
  `α·log α − loggamma(α)` constant through the Stirling remainder. Beta keeps the
  direct form for `φ ≤ 50` (its cancellation is small there) and uses a one-exp,
  `log1p` form (KL plus Stirling remainders) above. The Poisson `Hrr` is written
  without the `±½` cancellation. The `b`-free constants (`_nb2_group_const`,
  `_gamma_group_const`, `_beta_group_const`) are added once per group outside the
  inner solve.
- **Gamma warm start** at the log-mean-exp of `log y − η0` for the group. The old
  start could sit 26 to 31 below the optimum, where Newton creeps +1 per step.
- **Reporting.** Each fitter's `nll(θ, dec)` records every group's `ρ`. At `θ̂`
  the fitter calls `_va_inner_converged(dec, route)`, which warns and returns
  `false` if any `ρ > 1` or is NaN; the warning names the number of
  non-stationary groups, the largest `ρ`, and #965. `converged` is that AND
  `drm_optim_converged(res)`. The warning is silenced inside bootstrap refits by
  the `:drm_quiet_boundary` switch (`_without_boundary_warnings`), as the
  sparse-Laplace `_warn_at_clamp` is; the refit is still `converged = false`
  there.
- **`va_inner` keyword** on the five fitters (default empty) overrides the inner
  solve's `iters` / `maxiters`. Every inner call receives one `(iters, maxiters)`
  NamedTuple, built from the shared constants `_VA_POISSON_INNER_ITERS = 25` and
  `_VA_GHQ_INNER_ITERS = 30`, so an override shares compiled code with the
  default. The tests use it to force a non-stationary inner.
- **Docs.** The `Variational` docstring gains the runtime check.
  `docs/src/model-guides/marginal-la-vs-va.md` gains it and a worked Poisson
  example. The NEWS fragment is `news/965-va-inner-stationarity.md`.
- **Test** `test/test_va_inner_stationarity_965.jl`, 18 testsets:
  - unit cells for every family's inner solve;
  - the NB2, Gamma and Beta kernels against BigFloat, and the Binomial kernel against its Float64 softplus form and a closed-form value;
  - the envelope gradient, the damped and capped step, the reporter and the Newton zone;
  - six end-to-end fits that are wrong on base: Binomial, Poisson, two Gamma and Beta report `converged = true` at a non-stationary inner, and an NB2 fit reports a positive ELBO with `converged = true`;
  - a forced non-stationary inner in each of the five fitters, which must give `converged = false` with the #965 warning, quiet inside `_without_boundary_warnings`;
  - an NB2 no-false-alarm fit.

## 3a. Decisions and rejected alternatives

- **Iterate to tolerance and report**, rather than only warning. An inner point
  that is not stationary makes the reported ELBO lower than the profiled ELBO and
  makes the outer gradient (taken through the inner steps) wrong, so a warning
  alone leaves a wrong fit. The fixed unroll stays as the first steps of every
  solve; the extra steps, which ForwardDiff also differentiates through, converge
  the iterate the derivative is taken at.
- **Unit-free criterion.** `λ²` is twice the ELBO gain still on offer under the
  local quadratic model, so `1e-14` means the same thing for every family and
  group size. `|∇F|`, the measure the lane brief's suggested repro used, depends
  on the parametrisation and on group size. The absolute `1e-14` stops the solve.
  The relative bound only judges a group the solver cannot take that far (stuck,
  or out of iterations), since for a group ELBO of size `|F|` rounding alone is
  `~eps·|F|`.
- **The record is `converged` plus the warning, by design.** #965 asks that each
  fit record whether every inner profile met a stationarity criterion. `DrmFit` has
  no diagnostics slot, and adding a field would change a type shared by every
  route, so the record is folded into `converged` and the warning. The warning
  names the group count and the largest ρ. The per-group ρ stays available through
  the internal `fit.nll(θ, dec)`. A dedicated field and a `check_drm` entry are a
  follow-up (§5).
- **Fix the noise, not the tolerance.** The false alarms found while testing came
  from rounding noise in the group ELBO: NB2 near the Poisson limit, Gamma and Beta
  at large shape / precision, Binomial near saturation. The kernels were rewritten
  without cancellation and the Newton zone was added, so the check could stay
  tight. The old NB2 kernel also gave impossible positive ELBOs (+2577, +3539) on
  two Poisson-limit datasets.
- **The lane brief's suggested repro does not fail on base.** Issue #965 itself
  is a read-only finding and gives no repro. A Poisson `(1 | g)` fit with RE SD 3,
  G = 30, n_g = 5, seed 20261016 reaches inner stationarity on base (worst
  |∇F| = 1.2e-12 with MersenneTwister, 2.6e-12 with StableRNG). The red cases in
  the test file were found by scouting each family's failure regimes (stall, cap,
  roundoff) and then building end-to-end fits that hit them.
- Rejected:
  - raising the fixed unroll count, which does not fix STALL or ROUNDOFF and costs every fit;
  - a `|∇F|` threshold, which is not unit-free;
  - warning without iterating, which leaves the wrong fit;
  - putting the check in `optim_minimum_guard.jl`. It is not needed there, so this lane stays on f71b292 rather than #1066.

## 3b. Evidence

All runs are in this container: Julia 1.12.7 (conda-forge) unless stated, 4 cores.
"Base" is f71b292. "Fix" is this branch's working tree as each run used it; the
runs after the `va_inner` change are marked round 3.

**Red proof (round 3).** `test/test_va_inner_stationarity_965.jl` run against
base's `src/` gives 212 passed, 55 failed, 29 errored of 296, with at least one
failure or error in every one of its 18 testsets. Twelve fail on an assertion
about base's behaviour:
- the Poisson and Binomial inner cells;
- the Binomial kernel;
- the envelope gradient;
- the damped step;
- the NB2, Gamma and Beta inner cells;
- the Binomial, Poisson, both Gamma, Beta and positive-ELBO NB2 fits.

The other six fail because they call code base does not have: the stopped-short
ρ, the reporter, the NB2 kernel constants, the Gamma/Beta kernel constants, the
Newton zone and the `va_inner` keyword. On the fix the file gives 337 of 337 on
Julia 1.12.7 and 337 of 337 on Julia 1.10.4.

**All five families, checked one by one.** Base measurements are at the cells
the test file pins.

| Family | Failure on base (mechanism) | End-to-end on base | Fix |
|---|---|---|---|
| Poisson | all-zero group, σ_b = e⁴: \|∇F\| = 2.5 for any number of steps (STALL); \|∇F\| = 1.66 after 25 steps (CAP); huge Σe^{η0}: r climbs only at the \|Δr\| = 2 cap (σ_b ≈ 39), or creeps ~1e-2 per step under the old damping test (σ_b ≈ 1) | 7-observation fit, one all-zero group: `converged = true` at θ̂ = [37.599, 3.671], ELBO −32.366 | θ̂ = [2.8367, 1.4298], ELBO −24.8805 (7.49 nats higher); every inner \|∇F\| < 1e-6 |
| Binomial | \|∇F\| = 8.9e-2 (STALL), 0.60 after 30 steps (CAP), 5.4e-6 for any number of steps (ROUNDOFF in `s·η − n·softplus(η)`) | aggregated binomial, 1000 trials per row: `converged = true` with one group frozen at \|∇F\| = 2.5e-6 | every group \|∇F\| < 1e-6, every ρ ≤ 1 |
| NB2 | all-zero group, σ_b = e^4.25: \|∇F\| = 0.17 (STALL); kernel cancellation of `r·log r` terms near the Poisson limit | ELBO **+2577.42** with `converged = true` (crash-rate cell nb2-2.5-2-11, now a testset); ELBO **+3539.16** on `test_vcov_guard.jl`'s fixture (`converged = false` there); 19 of 108 crash-rate fits stop on a LineSearches `isfinite(phi_c)` assertion | −68.77 and −75.88, both `converged = true` (the fixture's largest inner ρ is 1.4e-18); 0 of 108 errors |
| Gamma | warm start ~26 below the optimum (m₀ = 0, m* = 26.3), Newton +1 per step: \|∇F\| = 0.15 after 30 (CAP); shape 0.01, σ_b = e⁵: \|∇F\| = 1.47 (STALL); CV ≈ 1% (shape 1e4): worst λ² = 9.4e-13, \|∇F\| = 9.7e-7 from kernel rounding | high-leverage group: `converged = true` at θ̂ = [2.084, 7.034, 0.457, 1.870], ELBO −230.693; the CV-1% fit is `converged = true` too | θ̂ = [1.948, 7.702, 0.379, 1.991], ELBO −225.239 (5.45 nats higher); CV-1% fit worst λ² ≤ 1e-14 |
| Beta | a positive-definite H passed the old det-only test: \|∇F\| = 2.9 (STALL); top quadrature node past `μ == 1.0` (old `1 − logistic(η)`): F = −Inf cliff, \|∇F\| = 12.3; at φ = 1e6 the old kernel misses the BigFloat value by more than the tolerance the new one meets | precision ≈ 1e6, small μ, 10 groups of 10: seed 5 reports `converged = true` with three groups frozen at \|∇F\| = 1.3e-5 to 4.1e-5 by kernel rounding. Seeds 4, 6 and 10 are red too, of the 11 seeds the scout tried; seed 1 is also `converged = true` with a group at \|∇F\| = 5.8e-6. θ̂ is right to ~1e-9, so the certificate was wrong but the estimate was not | seeds 4, 5, 6 and 10 `converged = true`, every ρ ≤ 2.8e-14, worst \|∇F\| 8.5e-12 (seed 5); θ̂ within 1e-9 of base |

**Identity harness** (no change where nothing was wrong). 120 fits: five
families × σ_b ∈ {0.3, 1, 2, 3} × n_g ∈ {3, 10} × 3 seeds, G = 25. 119 are
comparable; one NB2 fit errors on base.
- Over coordinates away from a flat boundary, max |Δθ| is 2.0e-10 (Poisson),
  1.02e-9 (Binomial), 5.0e-10 (NB2), 1.5e-10 (Gamma) and 1.8e-10 (Beta). Max
  |Δloglik| is 2.9e-11 and max |Δvcov| is 1.3e-10.
- Two RE log-SDs already at the zero-variance boundary moved (−20.5 → −19.8 and
  −21.1 → −19.3) with an unchanged ELBO.
- Two `converged` values changed. nb2-3.0-3-1 errors on base and converges on the
  fix. nb2-0.3-3-1 goes from true to false. Every inner ρ there is ≤ 3.4e-20, but
  the outer LBFGS stops at |g|∞ = 3.5e-8 against `g_tol = 1e-8` (base 4.1e-9). That
  component is the mean intercept's; the RE log-SD sits on its flat boundary
  (−19.3, base −21.1) with gradient ~1e-16.
- At the same θ the base and fix objectives agree to ~3e-14 there, so the flip is
  LBFGS taking a different path on rounding-level differences, not the inner
  check (audit re-run `audit/skeptic3/nb2_inner_diff.jl`;
  `final/nb2_samepoint_cmp.txt`).

**Crash-rate harness** (large RE SDs). 540 fits: five families × σ_b ∈ {1.5, 2.5,
3.5} × n_g ∈ {2, 4, 8} × 12 seeds, G = 20.
- Poisson, Binomial, Gamma and Beta: 0 errors on both, 108 of 108 converge on
  both, and max |Δθ| ≤ 3.5e-9.
- NB2: errors 19 → 0, converged 85 → 107. Its 3 `converged` changes without an
  error all go false → true, each with the NB2 `sigma` intercept at the
  Poisson-limit end (−8.6 to −12.0 on base, −20.6 to −23.3 on the fix) and an
  interior RE log-SD. The largest NB2 |Δθ| (13.8) is that flat `sigma` coordinate.
- One NB2 ELBO is lower on the fix: nb2-3.5-2-1, −50.8191 against −50.8176.
  Scored with the fix's cancellation-free objective, base's θ̂ gives −50.8191376
  and the fix's −50.8191320, so the fix's optimum is the better one. Base's
  −50.8176 was kernel rounding, and base's objective scores the fix's θ̂ at an
  impossible +2623 (`final/nb2_elbo_drop_{base,fix}.txt`).

**Cost and gradient.** `verify_grad_time.jl`: G = 40 groups of 8, per family.
Each time is the best of 3 warm fits, and the better of 2 alternating base/fix
rounds for `verify_grad_time.jl` (one round each for `beta_time.jl`). Suite shard
4 ran on another core throughout, and the evidence re-runs ran on a third during
all but the base `beta_time.jl` round.

| Family | Base | Fix |
|---|---|---|
| Poisson | 0.009 s | 0.011 s |
| Binomial | 1.11 s | 1.18 s |
| NB2 | 3.08 s | 2.49 s |
| Gamma | 2.67 s | 1.51 s |
| Beta (φ = 10) | 7.77 s | 7.61 s |
| Beta (φ = 1e4, `beta_time.jl`) | 16.7 s | 19.6 s |

- **Beta at large precision is slower.** It was about 17% slower in that one
  round; an audit re-run gave 21.7 s against 16.7 s.
- **The slower kernel is not the cause.** One kernel call and one objective
  evaluation are cheaper on the fix (0.044 s against 0.090 s per evaluation at θ̂).
- **The extra time is outer iterations and the final Hessian.** Measured at φ = 1e4
  (`final/beta_outer_{base,fix}.txt`):
  - LBFGS takes 19 iterations and 76 gradient calls on the fix, against 17 and 66 on base;
  - the final `ForwardDiff.hessian` at θ̂ takes 3.06 s against 0.88 s.
- **The gradient stays correct.** The outer ForwardDiff gradient matches a central
  finite difference (h = 1e-5) to ≤ 1.1e-7 at θ̂ and at a perturbed θ, where
  |g|∞ is 64 to 665. So the changed step rules do not break the derivative on
  these fits. No inner solve there ran past the fixed unroll (audit count), so
  this check does not exercise the extra steps.

**Full suite** (CI sharding, `DRM_TEST_SHARD=k/4`, `Pkg.test()` with
`--check-bounds=yes`, 372 files), on the code as of c560596.

| Shard | Files | Passed | Failed |
|---|---|---|---|
| 1 | 93 | 4,989 | 0 |
| 2 | 93 | 4,024 | 3 (all pre-existing, §4) |
| 3 | 93 | 4,273 | 0 |
| 4 | 93 | 3,652 | 0 |
| all | 372 | 16,938 | 3 |

In shard 2, `Pkg.test` aborted at the pre-existing `test_bridge_bootstrap_tree.jl`
failure after 14 files, so the other 79 ran one file per process.

The later edits are comments, docs, three testsets (Beta fit, NB2 positive-ELBO
fit, forced inner) and the `va_inner` keyword with its shared constants. Round 3
re-ran:
- the new test file (above);
- the 21 test files that call the `:VA` fitters or sit next to them: 21 files, 1,094 passed, 0 failed;
- the identity and crash-rate harnesses against the previous fix results: bit-identical, all 120 identity fits (θ, loglik, vcov) and all 540 crash-rate fits (θ, converged, errors).

**q=4 engine gate** (`bench/run_sparse_tmb_nd.jl`, PR template): logLik
−256.5064, converged, 114 iterations, identical on base and fix (the template's
reference is −256.51).

## 4. Pre-existing failures

Both fail identically on base f71b292 in this container (Julia 1.12.7,
conda-forge) and do not touch `:VA`:

- `test/test_bridge_bootstrap_tree.jl:28-29`. It aborted local shard 2, so the
  rest of shard 2 was run one file per process.
- `test/test_reml_q4_missing_response.jl:210` (`fit_reml.converged`; 18 passed, 1
  failed on both) in that per-file run. CI's Julia 1 shard 2 passed it on
  a56f657, where it runs after other files in one process.

## 5. Open / follow-ups

- **No dedicated inner-status record.** It needs a `DrmFit` field (or a
  diagnostics slot) holding the inner status (count of non-stationary groups,
  largest ρ), and a `check_drm` entry, so that an inner failure can be told apart
  from an outer one.
- **Outer `converged` at a variance boundary is fragile.** On variance-boundary
  `:VA` fits it is sensitive to rounding-level changes in the objective; the
  identity flip in §3b is one case.
- **Beta at large precision costs more.** The cost of the final nested-Dual
  Hessian there (3.06 s against 0.88 s) was not profiled.
- **The `:VA` clamp warning is not quieted in bootstrap.** That warning, in
  `_va_scored_sigma` (#959), still fires inside bootstrap refits.
- **The vcov step can still throw.** Gamma `:VA` at very small fitted scales
  (y·1e-60) and NB2 `:VA` with many all-zero groups can still throw
  `ArgumentError: matrix contains Infs or NaNs` there, as on base; some such NB2
  designs now get a pseudo-inverse vcov instead. That is vcov-guard territory, not
  inner stationarity.
- **`_logfactorial` is O(y).** It lives in `src/poisson.jl` and supplies the
  `log y!` constant the VA fitters use. It is harmless here but worth a
  `loggamma(y + 1)` switch for large counts.
- **Not tested in this lane:** `:VA` profile intervals, or how a profile refit at a
  non-stationary inner point reports it.

## 6. Rose audit

- **A workflow audit checked every claim against the evidence.** Four auditors, one
  per document group (user docs and code comments; test file; dev-log; PR body),
  extracted 334 factual claims and checked each against the code and the run logs.
  They flagged 67. An independent skeptic re-verified each flagged claim and
  cleared 7. A completeness critic checked the acceptance criteria, the Definition
  of Done, the licence and the callers of every changed function.
- **All 60 confirmed findings were applied.** They are in this report, the NEWS
  fragment, the PR body, the test-file comments and four code comments. The main
  ones:
  - two wrong numbers: the Gamma warm start (~26, not ~29) and the Beta scout's seed count (11, not 8);
  - the NB2 identity flip, which is on the intercept, not the log-SD;
  - the repro, which is the lane brief's, not the issue's;
  - the Beta switch comment, which said 3e-14 where the measured jump is up to 2.3e-13;
  - the Beta slowdown, whose cause is now measured;
  - the claim that "λ bounds the gradient error", now "controls" with the factor stated.
- **The critic's two substantive gaps are now testsets.** No test had shown that
  an inner failure turns a fit's `converged` false (now the forced-inner testset
  over all five fitters). NB2 had no fit-level red case (now the positive-ELBO
  testset).
- **Where the numbers come from.** Every number above comes from a run in this
  session on the code named, except the CI result on a56f657 in §4; the audit's
  own re-runs are under the session scratchpad's `audit/`. Nothing is
  extrapolated.
- **Licence and parity.** No drmTMB source was read or vendored, and the added
  lines contain no drmTMB or GPL text. `:VA` has no drmTMB counterpart: TMB is
  Laplace-only, so there is no R parity to state, and none is claimed. The `:LA`
  and `:Laplace` routes are not touched.
- **Scope.** Only `src/variational.jl` changed in `src/`. No bridge payload shape
  changed, and `optim_minimum_guard.jl` is not touched. The new `va_inner` keyword
  is internal (the fitters are not exported) and defaults to the old behaviour.
