## 1. Goal

Refresh the parity crosswalk with the current exact-head check states for five PRs and verify whether Totoro and DRAC can take additional work.

## 2. Implemented

- Added an exact-head snapshot to the existing PR crosswalk.
- Recorded Totoro load and the limits of the DRAC queue-status check.
- Left all PR branches, metadata, and compute queues unchanged.

## 3a. Decisions and Rejected Alternatives

- Did not start duplicate tests while owner-controlled exact-head jobs are queued or running.
- Did not submit work to Totoro because its measured load was about 301 on 384 CPUs.
- Treated Killarney's missing `squeue` and `sacct` commands as unknown queue status.
- Did not change draft state, submit reviews, contact owners, merge, release, or deploy.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-03-pr-followup-0011-utc.md`
- `docs/dev-log/check-log.d/2026-11-03-pr-followup-0011-utc.md`

The pre-existing modified Python bytecode file stayed unstaged and uncommitted. Its content was not compared with a pre-task copy.

## 5. Checks Run

- Rechecked live exact-head workflow runs for DRModels.jl #1214, #1215, and #1249, plus drmTMB #1500 and #1667.
- At 2026-11-03 00:11 UTC, #1214 CI was in progress and Documenter had passed; #1215 CI was queued and its Documenter run was cancelled; #1249 CI was queued and Documenter pending; #1667 R-CMD-check was in progress; #1500 R-CMD-check had passed.
- Queried Totoro using its existing ControlMaster socket: load average about 301 across 384 CPUs.
- Queried DRAC using the fixed read-only status wrapper: most host queries returned no job rows; Killarney lacked `squeue` and `sacct`.
- Ran `git diff --check`.
- No package tests, fits, simulations, installs, or Slurm jobs were run.

## 6. Tests of the Tests

Not applicable. Only an evidence document changed.

## 7a. Issue Ledger

- #1214, #1215, #1249, and #1667 remain owner-controlled with active or incomplete checks.
- #1500 has successful listed checks but remains a draft without formal review.
- No issue or PR state was changed.

## 8. Consistency Audit

- Matched each reported check run to the exact PR head from the live GitHub record.
- Recorded queued, pending, in-progress, cancelled, and successful states distinctly.
- Did not treat a successful documentation build as package-test success.
- Did not treat reachable compute sockets as available capacity or a ready R package environment.
- Style: 2/10, medium confidence. Genre/coverage: operational after-task report, read in full. Evidence/repair: section 11 said a missing scheduler command was “not an empty queue”; that contrast risks implying more than the query established, so it now says queue status is unknown. Gates: science = not applicable; facts = pass against the live PR metadata, workflow runs, and fixed host queries recorded above; references = not applicable because no cited sources appear. Provenance: self-review of this report, version 2, 2026-11-03; no held-out examples were used.

## 9. What Did Not Go Smoothly

The DRAC status wrapper could not query Killarney because `squeue` and `sacct` were absent from that shell's PATH. No queue conclusion is recorded for that host.

## 10. Known Residuals

- #1214 and #1249 checks remain live; #1215's current CI is queued and its docs run is cancelled; #1667 R-CMD-check remains live.
- Totoro should not receive CPU work until its load drops.
- The DRAC status check does not establish spare compute capacity or dependency readiness.
- This refresh does not establish R–Julia numerical parity, interval calibration, owner handoff, or merge readiness.

## 11. Team Learning

Reachability and capacity are separate measurements. The cluster status script can detect queued work when scheduler commands are available. A missing scheduler command leaves queue status unknown.

## 12. Cross-Product Coverage

Covers: current PR check state and compute endpoint reachability.

Does NOT cover: package-wide checks, numerical parity, recovery or interval calibration, owner handoff, merge, release, publication, or deployment.
