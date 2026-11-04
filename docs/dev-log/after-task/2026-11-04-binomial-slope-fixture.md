# Binomial slope fixture correction

## 1. Goal

Correct the public random-slope test fixture while retaining the evidence behind the failed CI check.

## 2. Implemented

The slope fit, predictions, bridge, and bootstrap use a separate dataset generated with group-specific random slopes. The intercept and Beta tests retain their original dataset. The convergence assertion remains in place.

## 3a. Decisions and Rejected Alternatives

Did not weaken convergence criteria or change the engine. Assumption: the public slope checks should use observations generated under a slope model; if the intent is specifically to test a zero-slope boundary, that needs a separate explicit test with boundary-aware expectations.

## 4. Files Touched

`test/test_binomial_sd_public.jl`; this report; `docs/dev-log/check-log.d/2026-11-04-binomial-slope-fixture.md`.

## 5. Checks Run

At base `f4fa3aed451636b6ffa442ce2632515513d23b1f`, the final test file was copied to a temporary `git archive` source snapshot and run with Julia 1.13.3 on macOS, `OPENBLAS_NUM_THREADS=1`, `JULIA_NUM_THREADS=4`. It passed 44/44 in 1m23.8s. The temporary manifest, copied from a matching project, has SHA-256 `1cb43b464b4e795d37658982b5df8d56f102bb8dcce902099de2ea33eaa72616`; its dependency versions need not match Linux CI. Test file SHA-256: `0dfb5d52dc7b83952c96c50fe5acbf90690e5d04474c1fb3712119ab5c47bc9d`. Command: `julia +1.13.3 --project=/tmp/drmodels-1252-repro-snapshot /tmp/drmodels-1252-repro-snapshot/test/test_binomial_sd_public.jl` with the thread settings above. `git diff --check` passed. On Totoro, a snapshot of commit `c1ec6c05fc88cfe6c86071da547c0a1ed898a457` passed the same test 44/44 in 1m40.8s on Linux with Julia 1.13.2 and the same thread caps. The source archive SHA-256 is `30e5fbb5f6616adee7a2efdac78bf862d3735ba5ab1feac252801bcdcae8b45a`; its generated Linux manifest is `bff4cf3fe2c4e8cdd5174b72afe388fae5a05f5e7564347d96fd5d2ea05aad3a`, and `focused.log` is `0e02dac61d5783e412748c57556ec988fa9bf62680446b5f38db829e77526a41`. The source and log are at `~/hsq_work/drm-slope-fixture-c1ec6c05/` on Totoro. The full suite has not been run.

## 6. Tests of the Tests

The original fitted slope model on data generated with random intercepts reproduced `converged=false` on macOS and gave a nearly singular Hessian, with group SDs from 3.07e-7 to 0.320. The same seed under a random-slope generating model gave `converged=true` and SDs from 0.515 to 0.784. The original Linux failure is recorded in [run 41842212229](https://github.com/itchyshin/DRModels.jl/actions/runs/41842212229). This is a single fixture comparison, not a general reliability or coverage study.

## 7a. Issue Ledger

PR #1252 remains blocked by its failed CI; this branch proposes an independent test-fixture correction. No issue was closed. The original boundary behavior remains a separate diagnostic question. The 2026-11-04 open-PR title/body search for binomial slope found no other PR proposing this fixture change; the search is not a full review of every PR diff.

## 8. Consistency Audit

Checked all slope uses in the modified test: the same new data are passed to the fit, prediction, bridge, and bootstrap. The unrelated malformed-loading and intercept checks retain the original data. Codex claim review found that the test data, response proportions, and predicted inputs are consistent, and the original convergence assertion is retained. No independent reviewer has yet reviewed this branch. The routed DRM.jl manifest requires preserving source identity and matching scientific claims to evidence; these shaped the source snapshot and limited claims. Golden Set was not run; no hub guard or cross-repository policy was changed.

## 9. What Did Not Go Smoothly

The original worktree had no resolved Julia manifest and could not load ForwardDiff. A matching project manifest was copied to a temporary snapshot; no manifest was added to the PR worktree. The original failure is platform-sensitive and the optimizer result was not retained for direct inspection.

## 10. Known Residuals

Only the focused test was run on the final candidate, on macOS and Linux. Linux CI and the full package suite remain outstanding. The local fixture comparison does not establish that the original boundary fit is incorrect. The full R–Julia parity goal, D-342, disclosed tolerance changes, and G5 calibration remain open.

## 11. Team Learning

A test that requires a positive random-slope fit should generate random slopes; a zero-variance boundary deserves its own explicit contract.

## 12. Cross-Product Coverage

The focused check covers this native Julia Binomial slope fixture, its prediction, bridge output and one-replicate bootstrap smoke. It does NOT cover R numerical parity, other families, parameter recovery, interval calibration, or full-suite behavior.

### Post-draft assessment

Style: 2/10, medium confidence. The report uses specific observations and records the limits of a single fixture. Science: pass for the stated comparison, not a coverage claim. Facts: pass against the local test output and recorded source identifiers. References: pass for the linked CI run. Reviewed by Codex on 2026-11-04.
