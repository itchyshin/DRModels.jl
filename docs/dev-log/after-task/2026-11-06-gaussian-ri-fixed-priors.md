# Gaussian random-intercept fixed-coefficient priors

## 1. Goal

Extend existing normal/ridge fixed-coefficient priors to Gaussian ML with one independent mean random intercept, keeping the group SD unpenalized. Draft PR #1258 carries Closes #1257; issue closure remains a merge action. No merge or release is included.

## 2. Implemented

The default exact Gaussian marginal route adds the prior once, preserves separate data likelihood and resolves native/bridge fixed-coefficient coordinates. Profile inventories reflect active-prior inference refusals. The shared variance-ratio entry now refuses fixed-prior confidence intervals through heritability, ICC and repeatability accessors, while preserving point summaries and other MAP-penalty behavior.

## 3a. Decisions and Rejected Alternatives

Retain the default exact route. Explicit marginal=:Laplace keeps its existing mean-random-effect refusal. Evaluate exact zero group variance using the separate boundary model. Native Julia and the bridge retained their original solver settings. The paired native R qualification uses its public multi_start=5 setting after retaining and independently diagnosing one inferior single-start minimum; no reference or threshold changed.

## 4. Files Touched

src/fixed_priors.jl, src/gaussian_core.jl, src/gaussian_ranef.jl, src/bridge.jl, src/introspection.jl and src/heritability.jl; two auto-discovered test files; the Julia numerical adapter; prior guide; NEWS/check-log fragments and this report. No GPL R implementation was copied.

## 5. Checks Run

Repaired engine sourceff10a2f9 passed all twelve native and twelve bridge fixtures against the frozen independent oracle. Native values cover all twelve points; derivatives cover nine positive-group-SD points. Three exact-zero points are value-only; arbitrary-point bridge derivatives are unavailable. Native RI tests passed48 expectations, including five real-factor selector checks. The derived-ratio repair passed23 expectations and43 existing affected checks. The guarded R live run passed87 expectations, including one actual bridge test, on the paired repaired source.

The original Julia1.10.12 Pkg.test runner completed all eight shards with exit0 and package-pass markers. The root independently verified the exact disjoint561-file union, with counts71 plus seven times70, and all eight JET gates2/2. Shard durations were1039,1168,1209,1377,1410,1259,1140 and8367seconds. Four Julia threads, one BLAS thread, parity suites off and the recorded dependency manifest were used. Complete raw receipts are retained in the twin R evidence directory guarded-julia-full/. Separate frozen-oracle and live-bridge evidence covers this slice; no full RCall parity campaign is claimed.

The numerical sparse baseline returned logLik -256.5059, converged=true,120 iterations and gradient residual7.36e-4 under its existing1e-3 tolerance. Its printed historical R constants do not provide a new paired performance measurement. Fresh source reviews and an independent receipt reviewer qualified the bounded local runtime evidence.

## 6. Tests of the Tests

Admission, missing-group preview and profile-readiness tests failed before their repairs. A later independent inventory exposed three ratio profile-interval failures, reproduced before the shared guard was added. Oracle mutation controls reject missing, doubled, misindexed and altered-constant priors. The zero-variance check does not take log(0).

## 7a. Issue Ledger

The isolated codex/gaussian-ri-priors-20261106 branch is pushed. Draft PR #1258 links drmTMB #1681. Both remain drafts and unmerged. Mathematical and runtime qualification includes the explicit baseline-kernel exception below. Final owned report and PR reconciliation records preserve the original failed attempts.

## 8. Consistency Audit

Fixed-prior metadata, separate likelihood, coordinate mapping, zero-ridge identities, update retention and affected inference refusals passed. The current docs-only source footprint matches repaired engineff10a2f9 for production and test code. Runtime receipt source, manifest, executable hashes and original runner commands are preserved. Independent source, runtime and kernel-control reviews support bounded qualification for draft review. Merge and release readiness are outside this report.

## 9. What Did Not Go Smoothly

Initial update-test syntax was corrected. The first inference inventory missed shared derived-ratio dispatch; its corrective review is retained. Boundary and singular-curvature warnings remain visible. Hosted stable-Julia CI failed an unpenalized binomial SD convergence assertion. The controlled baseline-kernel exception and its historical-attribution limit are recorded below.

## 10. Known Residuals

Original candidate Julia 1.13.3 CI retains two failed attempts at the unchanged unpenalized binomial slope convergence assertion. Controlled source/kernel comparisons independently establish a bounded baseline exception:

| OpenBLAS core | Pre-slice baseline | Prior candidate | Maximum gradient |
| --- | --- | --- | ---: |
| Cooperlake | 44/44 pass | 44/44 pass | 6.37e-9 |
| Haswell | 43/44, convergence flag false | 43/44, convergence flag false | 3.11e-7 |

Within each kernel, parameter bitstrings, objective, gradient, iterations and condition diagnostics match exactly between sources. Haswell reproduces the assertion and reciprocal condition 5.752216336949239e-17 seen in both original CI failures. The historical CI kernel was not recorded, so its hardware attribution is not proven. No assertion or convergence threshold changed, and no prior was added to this binomial fixture. The independent gate review accepts this existing baseline-exception path for prior-only draft qualification; it does not turn the failed CI attempts green. See the twin R evidence directory hosted-ci-baseline/kernel-pair/ and kernel-pair-outcome-review.md.

Both controlled 27-file prefixes passed, including all 44 target assertions and JET 2/2. The earlier dependency-wide coverage experiment retains two prefix timeouts; the corrected experiment used source-restricted coverage matching CI scope. The pre-slice hosted baseline passed all eight test jobs. R hosted qualification passed all three platforms. No calibrated interval coverage, posterior sampling, global optimization guarantee or general family parity is claimed.

## 11. Team Learning

The default Gaussian mean-intercept route is exact. Priors must enter the objective once; metadata wrapping is insufficient. Inference audits must include shared derived-quantity dispatch. Numerical qualification must retain source, runtime and solver limits. Memory receipt: the canonical contract and approved scope governed ownership and claims. No Golden Set, hub guard or memory file changed.

## 12. Cross-Product Coverage

Covers ML, complete data, unit weights and dense full-rank Gaussian fixed designs with one independent mean random intercept. Slopes, multiple/structured effects, SD/correlation priors, REML, other mixed families, interval calibration and arbitrary-point bridge Hessians remain deferred. Actual native factor selection was tested; numeric indicator bridge transport does not establish raw-factor bridge support. Both guide code blocks executed on the earlier candidate and were unchanged by the derived-inference guard repair. Documenter at385fde27 built successfully and published its PR preview; production-site deployment was skipped. Cleanup and R PR1651 remain protected. No merge, release or version bump occurred.
