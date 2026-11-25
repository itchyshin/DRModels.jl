# After-task: NB2 at the Poisson limit

Shannon; no subagents were running in the first round. Three reports, one root: with no overdispersion the
NB2 likelihood has no interior maximum in sigma, and the fitter treated the resulting plateau as an
ordinary optimum. Fixes: say so (warning naming the Poisson limit; `converged` untouched), survive the
line-search assertion (#1039), and look once for a higher interior optimum before accepting the plateau
(#1069). Decision recorded: the interior refit is a pragmatic multi-start, not a root-cause fix of why
the search left the interior; the cause on the plateau is untraced. The `-8` threshold is a judgement
(NB2 and Poisson logLik differ by far less than any tolerance beyond it), not a calibrated cut.

Second round (Code Reviewer block: the warning and refit covered only `_fit_negbin2` and
`_fit_negbin2_ranef`). Shannon, with read-only scout and review subagents (route catalogue, test audit,
warning plumbing; then an adversarial review). Findings that shaped the change, all measured locally
(Julia 1.12.6):
- The sparse-Laplace routes clamp the size at exp(8), so log sigma >= -4 there: the `-8` predicate never
  fired on them and the `-4` interior start sat on their clamp. Their threshold is the clamp.
- Past the clamp the objective is flat but the analytic dispersion gradient is the gradient AT the clamp
  (no chain-rule zero). LBFGS then stops at or drifts past -4 with `converged = false`. On simulated
  Poisson crossed designs this left fits on the clamp although an interior sigma was better: by 1.8 nats
  against the best fit on the clamp for the StableRNG seed 17 in the test file, and on 5 of 80 crossed
  draws in a seed search the refit was kept. On 12 phylo / relmat draws every refit returned to the
  clamp (no interior miss there).
- With `se = true` a discarded clamp fit would print a singular-Hessian warning, so the refit runs
  inside the spines before the covariance, `converged` and the clamp warning, never after.
- `src/optim_minimum_guard.jl`: `Optim.minimum` can be stale after a failed line search, which is the
  plateau case; both refits now compare the nll at each minimizer (this also changes the first round's
  GHQ refit).
- `Random.seed!` draws different counts on Julia 1.10 (seed 20282004: sum 165491 vs the issue's 165495),
  so the #1039 / #1069 data are fixtures; the seed-20282004 CSV has the md5 recorded in #1039.
- On the current engine seed 20282004 no longer reaches the Hager-Zhang assertion (LBFGS returns the
  plateau, log sigma -37.5); the test now pins what it does exercise: the refit to drmTMB's -5.049.

Adversarial review of the second round (four lenses, each finding put to an independent refuter):
- A REJECTED refit still moved the spines' shared inner-mode warm start (`last_b`), so the nll,
  gradient and finite-difference Hessian after it started from a different mode. At the clamp the
  Hessian's sigma row is tolerance noise, and SEs flipped (relmat: withheld NaN -> a falsely precise
  SE(log sigma) of 1.4e-8). Fixed: the refit restores the warm start when it is rejected. Checked
  bitwise against `ff0f9ee` (theta, logLik, converged, full vcov with `se = true`): identical for
  three rejected-refit clamp fits, a partly clamped covariate-sigma fit, the crossed AGHQ fit, an
  interior NB2 fit and two Gamma fits through the same spines; only the kept-refit fit differs.
- The clamp-route warning claimed the estimates and logLik "are those of NB2 at the capped size";
  at the clamp they can stop short of them (`converged` false), and location SEs can be NaN on the
  covariate-sigma route. Reworded. The docstring now names the routes that do neither.

CI on `edf46f4` (run 38905892545): Julia 1 shard 1/4 failed in this file. CI's `1` leg is Julia 1.13.2,
which had not been checked locally. Three assertions pinned a KEPT covariate-sigma refit
(`_nb2pl_phylo_z(4; size = 1500)`, to a partly clamped optimum 0.25 nats better). On identical data that
refit was kept on 1.12.6 and missed on 1.10.10 (fresh process) and 1.13.2, a floating-point knife
edge, so the pin was removed rather than loosened; the covariate-sigma spine keeps its end-to-end
warning tests and the unit-tested refit decision. Every remaining pin (crossed seed 17's kept refit,
the clamp-warning seeds, AGHQ seed 3) agrees on all three versions in fresh processes.

Known limitations, measured, not fixed here: at the clamp the other coordinates can be under-optimised
by the inconsistent gradient (up to 0.086 nats against the clamp profile on one phylo draw), the
reported sigma there is wherever the optimiser stopped (e.g. log sigma -9.1), and `converged` is often
false. The clean fix is a clamp-aware dispersion gradient or a polish at the clamp (#959 territory).
Out of scope: location SEs at the limit (SE path, #1073 stack); the NB2 routes listed in the check-log
as having neither the warning nor the refit.
Rose audit: pending.
