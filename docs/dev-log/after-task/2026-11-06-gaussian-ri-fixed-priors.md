## 1. Goal
Support normal and ridge priors on Gaussian fixed effects with one iid mean random intercept; retain the unpenalized group SD.

## 2. Implemented
Added the prior once to the existing exact marginal objective, retained separate data likelihood, and extended preview and bridge coordinate resolution for this model. Profile-target inventories now mark active fixed-prior MAP targets unavailable, consistent with the existing profile refusal.

## 3a. Decisions and Rejected Alternatives
The existing default exact Gaussian route is used. Explicit `:Laplace` retains its existing refusal for mean random effects. Assumptions: none beyond the approved scope.

## 4. Files Touched
`src/fixed_priors.jl`, `src/gaussian_core.jl`, `src/gaussian_ranef.jl`, `src/bridge.jl`, `src/introspection.jl`, `test/test_fixed_priors_gaussian_ranef.jl`, `tools/gaussian-ri-prior-adapters/julia-native.jl`, `docs/src/priors.md`, `news/gaussian-ri-fixed-priors.md`, this report and its check-log fragment.

## 5. Checks Run
On Totoro, Julia 1.10.12, four Julia threads and one BLAS thread: new tests 43/43 and existing introspection tests 21/21. On the earlier candidate source, existing prior tests passed 82/82 and four independent preflight fixture comparisons passed numerical thresholds for objective and identified coefficients, and point derivatives where log group SD is finite. The later source changes validate missing group labels during preview, correct an error message and correct profile-target readiness. Two preflight fits were near the group-SD boundary and produced singular-Hessian warnings. The final source-pinned native Julia and R-to-Julia bridge checks subsequently passed all twelve frozen fixtures against the independent oracle. Native pointwise values met the frozen limits on all twelve fixtures; gradients and Hessians met them at the nine positive-group-SD points, with value-only checks at the three exact-zero points. Fitted objectives and identified coefficients also met the limits. Bridge arbitrary-point derivatives remain unavailable. The broader 36-check aggregate also passed with native R using its recorded multiple-start configuration; its default ri_10 failure is retained. This evidence does not establish interval performance.

## 6. Tests of the Tests
The initial admission test failed on the old random-effect refusal before implementation. A later missing-group preview test also failed before the matching group-label guard was added. Profile-readiness tests failed for both active MAP model shapes before the inventory guard. The independent oracle has separate prior omission, duplication, index and constant mutation controls. Exact zero group SD was checked through the separate fixed-effect boundary model, not via `log(0)`.

## 7a. Issue Ledger
Closes #1257 when the pending checks pass and the change is integrated. Integration and publication remain with the coordinator.

## 8. Consistency Audit
Checked fixed-prior metadata and separate data likelihood, bridge coefficient coordinates, inference refusal, zero ridge, and prior retention in `update`. Route manifest lookup returned no LOAD-FIRST manifest for this worktree. No Golden Set was run; no hub guard or memory file was changed.

Memory receipt: the canonical contract, lane preflight and approved scope governed ownership, compute routing and claims; the route lookup had no worktree manifest.

Golden Set: not run; no hub guard was changed.

## 9. What Did Not Go Smoothly
The existing route drops missing responses before its usual design check, so the prior route now rejects incomplete responses earlier. The first expanded test used an incorrect `update` signature and was corrected before passing. A small varying-scale test produced a singular-curvature warning.

## 10. Known Residuals
The full suite remains pending. Both current prior-guide code blocks executed successfully on the pinned Julia source, including the random-intercept example. The full Documenter site has not been rebuilt by this slice. The twelve-fixture native and bridge grid and independent final source review passed at f60eb88e. Numerical raw results and the independent comparison are recorded in the twin R package under docs/dev-log/gaussian-ri-priors-20261106/final-numerical-evidence/. The closeout structure check passed, but the closeout compiler also found an unmet pre-existing `.unlazy/prior-timeout-validation-20261105/GATES.md` ledger; that ledger was not changed by this slice. The boundary warnings prohibit treating local curvature as a calibrated interval. No merge or release has occurred.

## 11. Team Learning
The `:LA` Gaussian mean-intercept route is exact; explicit `:Laplace` denotes a different, narrower route. Prior metadata wrapping alone does not add a penalty to an objective.

## 12. Cross-Product Coverage
Covers Gaussian ML fixed-coefficient priors with one iid mean random intercept and Julia bridge coordinate handling. Does NOT cover random slopes, multiple or structured effects, random-effect SD priors, REML, missingness, non-unit weights, other mixed-model families, interval calibration or arbitrary-point bridge Hessians.

Native factor follow-up: the current RI test file passed its existing 43 checks and five additional real-factor checks on Totoro Julia 1.10.12, four Julia threads and one BLAS thread. The exact `f: b` selector and its matrix column were checked with a successful fit and likelihood/prior decomposition. R-to-Julia validation of explicitly numeric indicator columns does not establish raw-factor bridge support; its existing refusal is retained.

The required `bench/run_sparse_tmb_nd.jl` numerical baseline ran on pinned f60eb88e with Julia 1.10.12, four Julia threads and one BLAS thread. It returned log likelihood -256.5059 (the documented -256.51 to two decimals), converged=true, 120 iterations and gradient residual 7.36e-4 under its existing 1e-3 tolerance. Its printed drmTMB time and likelihood are historical constants in the script, so its timing ratio is not a new paired performance or parity measurement. The raw log is retained under the pinned Totoro suite directory as sparse-engine-baseline.log.

A later independent post-fit inventory found that ratio profile intervals could still be returned for active fixed priors. The earlier inference audit missed this shared dispatch. The new entry guard refuses both interval methods through all three public accessors. TDD reproduced three profile failures; repaired tests passed 23 expectations, including point-summary and real other-MAP checks. Existing heritability/profile tests passed 43. Earlier numerical and full-suite receipts remain evidence for their original sources; fresh repaired-source qualification is pending.

## Repaired-source qualification checkpoint

The unchanged frozen oracle passed all36 route comparisons on R5d0f1596 and Juliaff10a2f9. The root independently reran all downloaded comparisons with exit0. Native R used the recorded uniform multi_start=5 policy; its earlier default local-minimum failure remains retained. Guarded live R tests passed87 expectations, including one actual bridge test. Three fresh completion reviews passed the bounded source gate. These results supersede the earlier candidate pins for numerical and inference-guard qualification.

The final full R check and Julia shard8 remain live. Seven final Julia shards exited0 and each reported that DRModels tests passed. Full-suite completion, final report reconciliation and PR preparation remain pending. No merge or release occurred.
