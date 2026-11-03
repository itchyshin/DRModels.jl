## 1. Goal

Add a deliberate optimizer choice to DRModels's existing `DRMControl`, on the primary ML routes qualified in the gate at `/tmp/cross-package-optimizers-20261018/gates/drmodels.md`. Keep defaults and the Gaussian random-intercept restart intact.

## 2. Implemented

`DRMControl(optimizer = :default | :lbfgs | :bfgs)` selects the two Optim.jl gradient methods directly. The 19 existing fixed-effect univariate `_ctl_optimize` call sites receive the selected method. Gaussian ML with one mean random intercept uses its own conditional two-start solve and records the method and each stage. The report attached to the fit records requested and actual method, native stopping output, attempted starts, a selected stage and a copied control snapshot. The former six-argument report constructor remains available and records its actual optimizer as `:unrecorded`. Explicit requests on unqualified models are refused. Fit-based profile and bootstrap entrypoints refuse an explicitly selected fit before a secondary optimization can silently change methods.

## 3a. Decisions and Rejected Alternatives

The selector stays inside `DRMControl`; a second public keyword would conflict with #1061. `:default` keeps LBFGS and the existing restart/tie rule. The Gaussian random-intercept route does not pass through the fixed-effect helper, because that would bypass its data-driven and boundary checks. The two added choices use Optim.jl's existing ForwardDiff path; Newton and derivative-free methods have no qualified route contract here. Assumption made without asking: the first random-intercept slice accepts only the optimizer and `se` controls, leaving `maxit`, supplied starts and multi-start unavailable on that route. A later owner decision could qualify those separately.

## 4. Files Touched

- `src/control.jl`, `src/gaussian_core.jl`, `src/gaussian_ranef.jl`, `src/inference.jl`
- `src/beta.jl`, `src/betabinomial.jl`, `src/binomial.jl`, `src/cumulative.jl`, `src/gamma.jl`, `src/lognormal.jl`, `src/negbinomial.jl`, `src/poisson.jl`, `src/skewnormal.jl`, `src/student.jl`, `src/truncated_poisson.jl`, `src/tweedie.jl`, `src/zeroonebeta.jl`
- `test/test_optimizer_selection.jl`
- `docs/src/model-guides/convergence.md`, `docs/src/capabilities.md`, `docs/src/drmtmb-parity.md`
- `news/optimizer-selection.md`
- `docs/dev-log/check-log.d/2026-10-18-optimizer-selection.md`, this report

## 5. Checks Run

At one Julia and one BLAS thread on Julia 1.10.0: `julia --project -e 'include("test/test_optimizer_selection.jl")'` passed 105/105 and printed `OPTIMIZER_SELECTION_PASS`; `julia --project -e 'include("test/test_lane_n3_drmcontrol.jl")'` passed 307/307; `julia --project=test -e 'include("test/test_ranef_varying_scale_convergence.jl")'` passed 6/6; `julia --project=test -e 'include("test/test_api_stability.jl")'` passed 200/200. The default coupled-Gamma bootstrap neighbor passed 15/15 in `test/test_locscale_bootstrap_refit.jl`, with 2/2 refits used in both serial and threaded calls and retained Hessian warnings. The RE panel retained 2 guarded throws and a boundary-Hessian warning, with zero reported-gradient violations. `git diff --check` passed. `julia --project=docs -e 'include("docs/make.jl")'` failed at `docs/src/reference/model-specification.md:143`, where the joint-imputation example asserted `is_converged(fit)`. The unchanged exact example failed with the same byte-for-byte log on archived PR-base `ef5decc7` using the same Manifest (SHA-256 `7e8224b5f2317c8ca694722ba1e3758f9e9a0b9eb3cd912dea94b2f09965d2e6`). Both logs are in `/tmp/cross-package-optimizers-20261018/gates/drmodels-{baseline,current}-example.log`. The full docs build was run on this branch only and remains red; it was not repeated after the inference guard repair.

PR #1136's first restacked CI run `39187798274` exposed a test-environment import error in Julia 1 shard 2 (job `117853924898`): the selector test directly imported `Optim`, which is not declared in `test/Project.toml`. The corrected test imports `Optim` through `DRModels`. A scratch environment copied from `test/Project.toml`, with this checkout developed locally, reproduced the direct-import error and printed `ISOLATED_OPTIM_IMPORT_PASS` for `using DRModels; using DRModels: Optim`. The focused selector file passed 105/105 again at one Julia and BLAS thread; logs are `/tmp/cross-package-optimizers-20261018/gates/drmodels-ci-import-isolated-green.log` and `drmodels-ci-import-focused-green.log`. These local checks do not replace the pending exact-head CI matrix.

## 6. Tests of the Tests

The first focused run failed because `DRMControl` lacked `optimizer`, with an unsupported-keyword error. An independent numerical review then found silent inference refits and a missing exported report constructor. The retained red test log at `/tmp/cross-package-optimizers-20261018/gates/drmodels-review-red.log` shows 17 failures on the old source: one six-argument `DRMControlReport` call and selected-fit profile/bootstrap refusals. The repaired focused log passed 105/105. A bad profile parameter and missing bootstrap data make any unguarded path fail cheaply; each test requires a selector-specific `ArgumentError`, proving refusal before a refit. A default-selected fit still profiles and bootstraps, and an explicitly selected fit still supports Wald intervals. The test also spies on the method object passed to `Optim.optimize`; `:bfgs` must receive an `Optim.BFGS` object and `:lbfgs` an `Optim.LBFGS` object. A flat-boundary objective forces the Gaussian specialized restart and its interior tie rule. Real small fits cover Gaussian fixed effects, Poisson, Gamma and Gaussian mean random intercept. These are optimizer-operation checks, not biological simulation or estimator-validity evidence.

## 7a. Issue Ledger

The change was built on `DRMControl` PR #1061, exact base `ef5decc7eef40cbc8b7f0f4b80623a9900dfd553`; that dependency was subsequently merged to `main` as `25964eef1e2709bc80c7fe4d2682cb6e1ceb1abe`. PR #1136 was restacked on that merged base without changing the selector's product source tree. Existing fixed-effect control scope is tracked by #948/#994. The previously noted stale `Optim.minimum` risk in multi-start ranking remains; this slice leaves that ranking as it was and reports native objective values as native, rather than claiming fresh objective evaluation there. No independent new algorithm or inference issue is closed.

## 8. Consistency Audit

All 19 `_ctl_optimize` callers were changed from fixed LBFGS closures to the helper's method argument. The preflight accepts only the ordinary Gaussian mean-intercept ML formula as a latent exception. The fit-based `bootstrap_ci`, `bootstrap_summary` and `bootstrap_result` methods guard both Gaussian and generic dispatch before simulation or refitting; `profile_result` and `confint(method = :profile)` guard before nuisance optimization. Wald intervals and default inference remain available. Neighboring `DRMControl`, Gaussian convergence and default coupled-Gamma bootstrap tests passed. The convergence guide, capability table, parity row, news and API docstring name the same qualified/refused routes. The random-effect file has older foreign refs with a separate convergence change near the fit flag; this branch did not replace that code.

## 9. What Did Not Go Smoothly

The first lease used an isolated-worktree registry key, which other DRModels lanes could not see; a second exact-path lease under the canonical DRM.jl key was granted and read back before continuing. The first red run was blocked by an unresolved Julia environment; local root/test/docs environments were instantiated. A bare `@formula` is not a `drm` method, so the forwarding test was corrected to the supported `bf(@formula(...))` call. The existing RE panel needs the `test/` environment for StableRNGs. The first restacked CI run found that the new selector test imported a package absent from the isolated test dependency list; the test now imports the same module through `DRModels` as neighboring tests do.

## 10. Known Residuals

Full `Pkg.test()` and exact post-repair CI have not been inferred from the focused checks. The joint-imputation example failed identically on the exact PR base and this branch; that local docs failure remains unresolved in this scope. Profile/bootstrap inference for explicitly selected fits is refused until the secondary optimizers can be forwarded and tested, not established by these checks. The selector does not calibrate intervals, improve identification or establish a faster optimizer. `Optim.minimum` on fixed-effect multi-start is still the prior native ranking value and may be stale after a failed line search. The Gaussian RE diagnostic recomputes its objective and gradient at each returned minimizer. The foreign convergence ref can create a later merge conflict in `src/gaussian_ranef.jl` and must be reconciled separately.

## 11. Team Learning

Memory receipt: loaded `route.py DRM.jl`'s LOAD-FIRST manifest and the repository AGENTS/HANDOVER routing. The route's reminder to avoid broad REML claims shaped the support matrix; exact-path canonical lane ownership prevented cross-lane ambiguity. Golden Set: not in scope; no known-mistake recovery or estimator claim was promoted. The repo source and bounded tests, rather than old memory, ground this report.

## 12. Cross-Product Coverage

Covers: direct `:lbfgs`/`:bfgs` primary ML optimization on the 19 fixed-effect univariate helper calls and ordinary Gaussian ML with one mean random intercept; `bf()` formula dispatch, selected-stage diagnostics, defaults, report-constructor compatibility, named preflight refusals and early fit-based inference refusals.

This slice does NOT cover REML, bivariate fits, structured/phylogenetic models, sigma-axis or multiple random effects, random slopes, alternate Gaussian integration/algorithm routes, profile/bootstrap optimizer forwarding, R bridge translation, inference calibration, performance or scientific recovery. `se = false` remains available on routes with no selector.
