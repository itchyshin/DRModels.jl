# After-task: block false plateau convergence (#944)

## 1. Goal

Give the next DRModels.jl agent a concrete record of the completed #944 slice. The
slice prevents a numerical plateau from being reported as convergence when Optim's
gradient criterion is still false.

## 2. Implemented

The slice added `drm_optim_converged` at
`src/optim_minimum_guard.jl:71`. The helper requires both
`Optim.converged(res)` and `Optim.g_converged(res)`. Reported convergence flags
on default-tolerance routes now use that helper.

The slice also added a focused regression test and an inventory of the routes
reviewed. This closeout adds the missing check-log fragment and after-task record.

## 3a. Decisions and Rejected Alternatives

The helper retains Optim's overall convergence check and adds the gradient
requirement. Replacing it with `Optim.g_converged` alone was unnecessary because
the public flag should still respect Optim's own result.

The deliberate `f_reltol` stops tracked in #946 were left alone. Those routes can
stop near a variance boundary while the gradient remains above `g_tol`, so applying
the #944 predicate there would change verified engine behavior.

No engine benchmark was run. NEWS was not updated.

## 4. Files Touched

The completed implementation commit modified or added:

- `docs/dev-log/evidence/2026-10-04-sweep-blocked-inventory.md`
- `src/betabinomial.jl`
- `src/binomial.jl`
- `src/bivariate_student.jl`
- `src/cumulative.jl`
- `src/gaussian_core.jl`
- `src/gaussian_locscale_phylo.jl`
- `src/gaussian_lss.jl`
- `src/gaussian_meta.jl`
- `src/gaussian_ranef.jl`
- `src/gaussian_sparse_lss.jl`
- `src/gaussian_structured.jl`
- `src/joint_missing_finite.jl`
- `src/joint_missing_predictor.jl`
- `src/joint_missing_two_predictor.jl`
- `src/locscale_corr.jl`
- `src/locscale_sigma.jl`
- `src/lognormal.jl`
- `src/mixed_family.jl`
- `src/optim_minimum_guard.jl`
- `src/phylo_interaction.jl`
- `src/poisson.jl`
- `src/skewnormal.jl`
- `src/student.jl`
- `src/temporal.jl`
- `src/truncated_poisson.jl`
- `src/tweedie.jl`
- `src/variational.jl`
- `src/zeroonebeta.jl`
- `test/test_sweep_blocked_944.jl`

This closeout adds:

- `docs/dev-log/check-log.d/2026-10-04-sweep-blocked-944.md`
- `docs/dev-log/after-task/2026-10-04-sweep-blocked-944.md`

## 5. Checks Run

Totoro evidence from 2026-10-04 records
`test/test_sweep_blocked_944.jl` passing 6 of 6 tests in 1.6 seconds with
Julia 1.13, one OpenMP thread, one OpenBLAS thread, and four Julia threads.

The engine benchmark was not run. This closeout did not rerun the test.

## 6. Tests of the Tests

The regression test constructs a plateau result for which `Optim.converged` is
true while `Optim.g_converged` is false. It checks that
`drm_optim_converged` rejects that result. A normal LBFGS result checks the
opposite case, where both Optim checks and the helper return true. The old
plateau behavior would fail the new rejection assertion.

No separate mutation run was recorded.

## 7a. Issue Ledger

- #944: addressed by the implementation and focused regression test.
- #946: deliberate `f_reltol` routes deferred and unchanged.
- [PR #1041](https://github.com/itchyshin/DRModels.jl/pull/1041): still a
  draft at `b66aafc25`; not merged as of 2026-10-04.

## 8. Consistency Audit

The committed inventory records a sweep of the same reported-flag pattern. It
switched 67 call sites to the shared helper, then restored the two mixed-family
control lines because they are not reported convergence flags. It separately
lists deliberate `f_reltol` stops, profile retries, Laplace outer loops, and
files owned by other open pull requests.

The check-log fragment follows `docs/dev-log/check-log.d/README.md`: one file,
one five-column row, and no table header. The frozen
`docs/dev-log/check-log.md` was not edited.

Memory receipt: the repository rules, the check-log README, the committed sweep
inventory, and the hub after-task protocol shaped this record. The repository
has no `docs/design/10-after-task-protocol.md`, and `route.py` found no
DRModels.jl LOAD-FIRST manifest. Golden Set: not checked because this closeout
adds prose only and does not change the known-mistake mechanism.

## 9. What Did Not Go Smoothly

The repository-specific after-task protocol named in the task was absent. The
hub protocol supplied the report structure instead. No implementation problem
was investigated during this closeout.

## 10. Known Residuals

The focused test is the only recorded test run for this slice. There is no
engine benchmark result and no full-suite result in this record. The #946
`f_reltol` routes remain outside the helper by design. PR #1041 remains a draft,
and NEWS remains unchanged.

## 11. Team Learning

`Optim.converged` is an OR across the x, objective, and gradient stopping
criteria. A reported fit flag that means "the gradient criterion passed" must
check `Optim.g_converged` explicitly. Keep that stricter interpretation limited
to default-tolerance routes; objective-relative stopping near a variance
boundary has a different contract.

## 12. Cross-Product Coverage

Coverage includes reported convergence flags on the default-tolerance routes
listed in the committed sweep inventory, plus a direct LBFGS plateau
counterexample and a normally converged LBFGS control.

This slice does NOT cover the #946 `f_reltol` routes, profile retry decisions,
Laplace outer-loop stopping, files owned by the other open pull requests listed
in the inventory, full-suite behavior, performance, or engine benchmark results.
