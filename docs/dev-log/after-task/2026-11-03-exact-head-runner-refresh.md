# After-task: exact-head Julia runner refresh

## 1. Goal

Refresh the evidence crosswalk for DRModels.jl #1214 and #1249 from current PR, workflow, job, and raw failure-log evidence.

## 2. Implemented

Added a timestamped receipt to the existing PR review crosswalk. It records the exact heads and base, terminal and live shard states, the two #1214 failure causes, and the unverified status of #1249's candidate fixes.

## 3a. Decisions and Rejected Alternatives

- Kept #1249's changes provisional pending its completed exact-head test run.
- Did not launch a duplicate run on Totoro or DRAC because #1249's exact-head Julia suite is already running.
- Did not update the separate dirty crosswalk worktree or infer its owner from the branch name.
- Did not change PR state, source code, test expectations, or remote branches.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md` (modified)
- `docs/dev-log/after-task/2026-11-03-exact-head-runner-refresh.md` (created)
- `docs/dev-log/check-log.d/2026-11-03-exact-head-runner-refresh.md` (created)

## 5. Checks Run

- Refreshed live PR metadata for #1214 and #1249, including exact heads, bases, states, and draft status.
- Refreshed workflow-job status for CI runs `41539113997` and `41548951015` and Documenter run `41548951145`.
- Retrieved logs for #1214 failing jobs `125614277806` and `125614277774` and verified the failing test names, expressions, observed values, and source locations.
- `git diff --check`: passed.
- `Rscript ~/shinichi-brain/tools/check-after-task.R docs/dev-log/after-task/2026-11-03-exact-head-runner-refresh.md`: passed.
- `python3 ~/shinichi-brain/tools/slop_check.py <absolute after-task path>`: zero findings. The initial relative-path invocation resolved against the brain root and could not find the report; rerunning with its absolute path completed the check.

## 6. Tests of the Tests

No local tests were run. The cited exact-head CI jobs are current remote evidence. The #1214 failure log confirms the failing assertions; #1249 has not completed its full Julia suite, so its fixes remain unverified.

## 7a. Issue Ledger

- No issue was opened or closed.
- #1214 and #1249 remain open. No review, comment, merge, or draft-state change was made.

## 8. Consistency Audit

Compared the current exact-head logs with the existing crosswalk entry. #1214 shard 1 fails the stored parameter-vector comparison at `1e-8` with maximum difference about `1.82e-8`; shard 4 fails the three boundary expectations because the fit is reported converged and its covariance is finite. #1249 changes both relevant areas but all four Julia shards remain in progress at the recorded refresh. The receipt therefore does not say the failures are fixed.

## 9. What Did Not Go Smoothly

The failure logs were not exposed in the first job-status response; a second targeted retrieval was needed to read the raw job output.

## 10. Known Residuals

- #1214 shard 3 is still running, so its CI run is not terminal.
- #1249's four Julia shards are still running, so the proposed fixes have no completed exact-head suite result.
- Ownership and disposition of the separate dirty crosswalk worktree remain unresolved.
- No package tests, fit, simulation, or compute-host job was launched in this slice.

## 11. Team Learning

Use exact workflow job logs to attribute failed assertions before updating a crosswalk. Do not start a second run while an owner-supplied exact-head run already exercises the candidate fix.

## 12. Cross-Product Coverage

Covers: current evidence attribution for two Julia documentation PR candidates.

Does NOT cover: merge readiness, broad R–Julia parity, the R package, CMR, interval calibration, or package release readiness.

## 13. Prose assessment

Perceived AI-like style: 2/10, medium confidence; concise operational prose and exact identifiers suit this evidence receipt. Genre and coverage: self-review of the full after-task report. Evidence and repair: the fixed headings are repetitive by project requirement; no further repair is needed. Gates: science not applicable; facts pass against the exact PR metadata and job logs named above; references not applicable because the report makes no literature claims. Provenance: self-review of this saved draft on 2026-11-03; comparable after-task reports were read during this task.
