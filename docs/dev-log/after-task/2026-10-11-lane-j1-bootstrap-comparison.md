# 2026-10-11 — lane J1: bootstrap, likelihood-ratio tests, model comparison

Shannon, with Noether (bootstrap status contract), Hopper (bridge payloads and
LR guards), and Rose (claim versus evidence). No spawned subagents.

Branch `claude/lane-j1-bootstrap-comparison`, stacked on PR #1044
(`cursor/triage-g-twin-inputs`), which already fixes #1018, #1008 and #962 and
adds the nobs / marginal / MAP guards to `lrt_boundary`. Several assigned
issues (#1046, #1048, #1051, #1052, #1053) are follow-ups filed from #1044 and
target code that exists only on that branch (`src/ci_args.jl`).

## What changed

- **#1036.** `_marginal_compare_guard` no longer lets two `:VA` fits through
  on the equal-tag shortcut. `lrtest`, `anova` and `lrt_boundary` refuse the
  pair; `aic` / `bic` / `aicc` already did.
- **#1053.** `_nobs_compare_guard` also compares the observed response values
  stored on the fit (`fit.obs`, non-finite entries dropped, sorted). Equal
  counts on different rows are "different samples". Sorting keeps a re-ordered
  copy of the same rows comparable. Limitation: if the rows that differ carry
  identical response values (ties are common for counts), the check cannot see
  the difference.
- **#1002.** `lrt_boundary` requires `dof(fit_full) > dof(fit_reduced)` (a
  swapped call used to return p = 1 silently) and `q ≤` that difference. Both
  `lrtest` and `lrt_boundary` warn when either fit reports `converged = false`.
  The family check is left for a decision (Poisson against NB2 is a common,
  legitimate boundary test).
- **#1046.** `_bootstrap_boundary_params` unpacks each `:recov`
  (`[log L11, log L22, L21]`) and `:phylocov` (log-Cholesky) draw into SDs and
  correlations and applies the scalar cutoffs. A hit flags every entry of the
  block.
- **#997.** `_marginal_simulator_build` refuses an ordinary `(0 + x | g)` or
  `(1 + x | g)` bar (it drew intercepts only) and reports, via `@warn`, every
  remaining reason it falls back to the conditional `simulate`. The two bare
  `catch` blocks around `predict` now report the error.
- **#1004.** `_q4_cor_offdiag` returns `NaN` for a zero-SD axis. Percentile rows
  use finite draws only. A nonzero ridge in `_q4_re_prior_chol` warns and is
  returned as `prior_ridge`. The missing warning (item 1) was fixed by #1044.
- **#1051.** `_mf_boot_ci` returns a status using `_finish_bootstrap_status`;
  `fit_mixed_family` returns `boot_status` and `n_boot_failed`. The existing
  `max(10, B ÷ 2)` rule is unchanged and now reports `bootstrap_unavailable`.
- **#1011.** Items 1 and 2: the profile-failure message no longer recommends
  `method = "wald"`, and the q2 bridge entry points reject unknown option keys.

- **#951 (added by the coordinator).** `_bridge_fit` checks every forwarded
  keyword with `hasmethod` against the family's `drm` method and refuses an
  unsupported one by name (a covariance provider gets a "no structured route"
  note). Before, only `marginal` was checked; the rest reached R as a bare
  `MethodError`. The Gaussian fixed-effect half (`algorithm = :gls`/`:lbfgs`/
  `:sparse` accepted with no effect) is in `gaussian_core.jl`, lane J3's
  area, and the bivariate q4 `g_tol` comment is a direct-Julia-call issue
  outside the bridge; neither is changed here.

- **#1052 (decision 18, approved 2026-10-11).** The public `bootstrap_ci`,
  `bootstrap_summary` and `bootstrap_result` methods (and `_bootstrap_result`)
  default to `failures = :skip`: a refit that throws is recorded in
  `failures`, dropped, and the run returns `bootstrap_incomplete` with a
  warning, as drmTMB does. `failures = :error` still aborts; all replicates
  failing is still an error.

## Not changed (owner decisions)

- #1011 item 3: a minimum-success floor for status `"bootstrap"`.
- #1025: forwarding is already on main for fit-based calls; formula-based
  bootstrap methods have no `method` / `marginal` keyword (API addition).
- #1048: blocked on #1047 (J2, EM convergence flag).

## Verification

Julia could not be installed in the authoring container (network policy), so
nothing was run locally. The PR's GitHub Actions CI is the test run of record.
