## 1. Goal

Refresh the exact-head reviews, CI states, and compute-resource readiness for the active DRM parity cleanup without duplicating active checks.

## 2. Implemented

- Added current exact-head status for DRModels.jl #1249, drmTMB #1667, and drmTMB #1500 to the parity crosswalk.
- Recorded two independent source-review findings and the decision not to run duplicate CI or start a cluster job.
- Checked Totoro load and existing DRAC connections. Fir is reachable, but its base R module lacks the core package-check dependencies.

## 3a. Decisions and Rejected Alternatives

- Kept all PR branches and metadata unchanged. Both reviewers were asked to work read-only.
- Did not run package checks because the exact PR checks are already active and Fir lacks required packages. Did not install dependencies or submit a Slurm job.
- Did not retry Rorqual after its existing socket returned an MFA-required error, and did not open a new login.
- Assumption: “Totoro and DRACs are back” permits checks through existing authorized connections. It does not approve a new compute campaign. If this assumption is wrong, a bounded task can be selected after exact CI finishes.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-parity-resource-ci-refresh.md`
- `docs/dev-log/check-log.d/2026-11-02-parity-resource-ci-refresh.md`

The pre-existing modified bytecode file `tools/__pycache__/runtests_files.cpython-313.pyc` remains untouched.

## 5. Checks Run

- `bash ~/shinichi-brain/tools/lane_preflight.sh "$PWD" --file docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`: no competing ref had missing work on the path; 28 lanes were live, so this lane was named and a path-scoped lease was obtained.
- Read-only GitHub metadata and workflow jobs: #1249 CI pending and docs queued; #1667 source-tree job in progress and OS matrix queued; #1500 shards 1, 2, and 4 passed and shard 3 in progress.
- Read-only host checks: Totoro load about 296/384; Fir load about 11 with zero queued jobs; Rorqual existing socket rejected authentication because MFA is required.
- Loaded Fir R 4.6.1 and checked packages: `cli`, `lifecycle`, `TMB`, `RcppEigen`, `testthat`, and `pkgdown` absent.
- No package tests, model fits, simulations, benchmarks, installs, or Slurm jobs were run.
- `git diff --check`, the after-task validator, and the prose check were run after writing this report.

## 6. Tests of the Tests

Not applicable. This was an evidence and resource-readiness update. No test code changed.

## 7a. Issue Ledger

- #1249: open. Review identified two P2 follow-ups; exact-head CI is pending.
- #1667: open. Review identified a P2 warning-visibility concern; exact-head CI is running.
- #1500: open draft. Three of four Ubuntu shards passed; one remains active.
- No issue or PR metadata was changed.

## 8. Consistency Audit

- Compared each recorded SHA with live PR metadata before associating the workflow runs.
- Kept native or numerical parity, interval calibration, and release claims outside the scope of these code reviews and CI states.
- Distinguished a reachable host from a usable package-check environment. No cluster job was submitted from a login node.
- Style assessment: 2/10 perceived AI-like style, medium confidence. Genre and coverage: this short operational record, read in full. The exact-SHA bullets and explicit compute decision are concrete and non-repetitive; no style change needed. Science: not applicable. Facts: checked against live PR metadata, run jobs, and host output. References: not applicable beyond direct run links. Provenance: self-review, this report version, 2026-11-03, no unrelated examples used.

## 9. What Did Not Go Smoothly

The first `closeout.py new` call resolved a relative output path inside the brain repository instead of this package worktree. That generated template was removed immediately. The absolute-path `closeout.py check` also failed because the helper invokes the brain repository's validator against an external package report. The package-specific after-task structure validator passed directly, and this limitation is recorded rather than treated as a package failure.

## 10. Known Residuals

- The PRs remain owner-controlled. The P2 findings are not fixed by this lane.
- Fir needs a staged dependency library before it can run R CMD check. Package installation time has not been measured.
- Totoro load remains high. The Rorqual socket needs the user to restore MFA access before it can be reused.
- Current exact-head CI may change after this observation; recheck only after its status changes or the normal backoff.

## 11. Team Learning

Memory receipt: read the finite-work and compute-routing instructions, ran lane preflight, and inspected the latest crosswalk receipt. The route helper found no LOAD-FIRST manifest for this generated worktree. These instructions shaped the exact-head recording, the path-scoped lease, and the decision not to duplicate active CI.

Golden Set: not run; no code or test behavior changed.

## 12. Cross-Product Coverage

Covers: exact-head review and CI status for three parity-related PRs; read-only readiness of Totoro and selected DRAC hosts.

Does NOT cover: R-Julia model parity, CMR, interval coverage or calibration, full package checks on Fir, PR ownership, merge, release, publication, or deployment.
