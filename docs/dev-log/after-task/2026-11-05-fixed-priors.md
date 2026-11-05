## 1. Goal

Implement and check the approved first fixed-effect normal-prior and ridge
slice for DRModels.jl, including the drmTMB bridge.

## 2. Implemented

The experimental constructors and resolver select exact final fixed-effect
columns. Fixed Gaussian location and log-scale coefficients and binomial-logit
location coefficients receive the penalty in their objective. Active priors
produce MAP fits with separate data likelihood and prior contribution; zero
ridge remains ML. Bridge payloads check R and Julia selections before fitting.
Unsupported inference and model routes are refused; summary tables withhold
nominal tests and intervals.

## 3a. Decisions and Rejected Alternatives

Normal density constants are included; ridge uses exactly `lambda * beta^2 / 2`.
Random effects, missing rows, non-unit weights and REML are outside the first
slice. Local penalized curvature is retained but is not presented as an
automatically calibrated interval. No new general separation test was added:
partially penalized binomial fits warn that all separating directions need
regularization. Assumptions made without asking: none beyond the approved plan.

## 4. Files Touched

`src/DRModels.jl`, `src/fixed_priors.jl`, `src/gaussian_core.jl`,
`src/binomial.jl`, `src/bridge.jl`, `src/comparison.jl`, `src/inference.jl`,
`src/summary.jl`, `src/visualization.jl`, `src/bias_correct.jl`,
`test/test_fixed_priors.jl`, `test/test_api_stability.jl`, `docs/src/priors.md`,
`docs/make.jl`, `news/fixed-priors.md`, this report and the matching check-log
fragment. No source was copied from the GPL R implementation.

## 5. Checks Run

On Totoro, Julia 1.10.10, four Julia threads and one BLAS thread, the frozen
predecessor source aggregate `d8fad19a2bbf7cac678ac2a0c7a28376f850e8efecd7a88e91c0cf55da6cbfdd`
passed 79 focused assertions, including exact ridge, independent objective and
derivative checks, bridge agreement, selection, guard and update tests. The
augmented log is `evidence/julia/focused-augmented.log` under
`/home/snakagaw/hsq_work/drm-priors-20261105-01a1ad0b/`. Fourteen affected
Gaussian, binomial, summary, comparison, inference, bridge, control,
visualization, bias-correction and API test files passed (exit 0), in
`evidence/julia/affected-frozen.log`. Two predecessor `Pkg.test()` runs on
`efd2d993` and `d8fad19a` were superseded and stopped with SIGTERM (groups
1397202 and 1456256; both exit 143, no remaining group members); their partial
logs are retained and they are not passes. The final source aggregate is
`a30e347285f925af2e128ed6350fafe4d6b9035668444606bfa2fcea13e26653`;
its full suite is running. Earlier-source results cannot establish a full-suite
pass on the final source. The final focused suite passed 82/82 and the 14-file
affected suite exited 0; see `evidence/julia/focused-final2.log` and
`evidence/julia/affected-final2.log`. Exact source and test hash manifests are
`evidence/julia/source-a30e3472.sha256` and
`evidence/julia/tests-a30e3472.sha256` under the same Totoro directory.
The new guide example ran successfully on a30e, and a minimal one-page
Documenter HTML build completed with a nonempty `priors.html`; see
`evidence/julia/prior-doc-final.log`. Missing Git-remote and landing-page
warnings are expected for that disposable one-page build.

## 6. Tests of the Tests

Constructor, Gaussian-route and bridge tests were run before production wiring
and failed on their respective missing constructor, unsupported keyword and
unknown bridge option. Independent analytic and central-difference oracles
check the fitted objective rather than relying on agreement between engines.
Formal mutation checks are not yet run.

## 7a. Issue Ledger

No external issue was created or changed. Integration and mathematical review
found inference-table leaks, non-unit weights, zero-ridge geometry and update
switching; prior-specific changes and tests address them. Final independent
review and the full-suite result remain open.

## 8. Consistency Audit

The source was checked for normal CI, profile, profile-surface, bias-correction,
bootstrap, comparison, prediction-SE, simulation and summary entry points.
The bridge retains legacy behaviour for unrelated phylogenetic MAP fits. The
site menu links the prior guide, and new exports are classified experimental.

## 9. What Did Not Go Smoothly

The first Totoro project lacked a resolved manifest; its isolated project was
instantiated before tests. Initial R JuliaCall startup failed before model code
without the existing bundled-libunwind launch workaround. A hub closeout
utility wrote a placeholder outside the worktree; that newly generated file
was immediately removed. No existing hub content was changed. The first
isolated documentation build failed because its copy had no Git remote. The
retry disabled remote links in the disposable copy only, then failed the
convergence assertion in `docs/src/reference/model-specification.md:166`.
The exact example also failed on clean `24ac6ca` with Julia 1.10.10 and the
same docs manifest (SHA-256 `fb61aa4a8fdd46ef3c84709839ec707493e3441c395cdd56a48d398697143641`).
This is a matched example-level baseline, not a full clean-main site build.

## 10. Known Residuals

The final source has not passed a full package suite. The predecessor-source
Documenter retry failed as above; the final full documentation build remains
unverified despite the successful new-page check.
Broader R–Julia parity, interval calibration, random-effect SD priors,
and sampling are not established. The exact partial-prior binomial existence
problem is not generally solved. Source is uncommitted, unpushed and unmerged.

## 11. Team Learning

Memory receipt: the DRM.jl load-first manifest, project instructions and
validation-harness shaped source isolation, independent oracles and the
likelihood-scale checks. Golden Set: not run in this bounded implementation;
no durable brain write was made. A penalty added to the objective also changes
every inference and rebuilding surface, so those surfaces need an explicit
audit rather than a single `confint` guard.

## 12. Cross-Product Coverage

Covers: fixed Gaussian `mu`/`sigma`, fixed binomial-logit `mu`, selector
resolution, likelihood/objective/gradient separation, MAP reporting, selected
refusal paths and the bounded bridge fixtures.

This does NOT cover random effects or SD priors, non-unit weights, missing
responses or predictors, REML, other response families, posterior sampling,
coverage calibration, general parity, release or deployment.
