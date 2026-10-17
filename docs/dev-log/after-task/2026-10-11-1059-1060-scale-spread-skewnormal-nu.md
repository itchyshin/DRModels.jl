# After-task: #1059 / #1060 (2026-10-11)

Shannon; no subagents running.

- **#1059.** The issue's own repro (constant response in one group, `sigma ~ x`) already
  gives `fit.converged = false` on main (the gradient criterion fails), so the
  headline "reports converged = true" did not reproduce for it. The real gap was
  `_nondegenerate_fit`, which tested only `max(sigma)`: a fit with one group's sigma
  collapsed and another's O(1) passed `is_converged` if the optimiser flag was true.
  Added a unit-free spread bar (min/max fitted sigma > 1e-8).
- **#1060.** `SkewNormal` printed Wald SEs for `nu` with no caveat. On symmetric data
  nu-hat is routinely about ±1 (SE 0.4 to 1.4), so a bar on nu-hat alone misses it; the
  warning fires on max |nu| < 0.5 or a 95% Wald test of the `nu` block not rejecting 0.
  The threshold is a heuristic.
- **Not done (outside this repo).** drmTMB `is_converged.drmTMB_julia`,
  `convergence_status.drmTMB_julia` and the `skew_normal_nu_wald` row in
  `check_drm.drmTMB_julia`.
