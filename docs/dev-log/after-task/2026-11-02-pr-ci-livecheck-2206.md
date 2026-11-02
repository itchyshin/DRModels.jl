## 1. Goal

Refresh exact-head checks after a changed drmTMB #1657 run and keep the current parity evidence available to the next lane.

## 2. Implemented

- Added a 22:06 UTC status receipt to the parity PR crosswalk.
- Confirmed #1657 run `41522911628` passed its OS matrix and now has Windows, macOS, Ubuntu, and source-tree jobs running.
- Confirmed #1249 shards remain in progress and #1659 Windows remains cancelled.

## 3a. Decisions and Rejected Alternatives

- Left PR source and metadata unchanged because the branches are owner-held and no review or merge authorization was given.
- Did not run computation on Totoro or DRAC. The changed evidence is CI status, which does not require local model work.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-pr-ci-livecheck-2206.md`
- `docs/dev-log/check-log.d/2026-11-02-pr-ci-livecheck-2206.md`

An unrelated modified Python bytecode file remains untouched.

## 5. Checks Run

- Exact-head GitHub diagnostics were read for #1249, #1657, and #1659 at 22:06 UTC.
- `git diff --check` passed.
- The after-task structure checker passed.
- `slop_check.py` reported 0 findings and 0 em dashes for this report.
- No package tests, fits, simulations, or benchmarks were run.

## 6. Tests of the Tests

The GitHub checks recorded here apply to the three named PR heads. This documentation update was checked separately with structural and whitespace/style checks.

## 7a. Issue Ledger

- No issue or PR metadata changed.
- #1249, #1657, and #1659 remain open.

## 8. Consistency Audit

- The new #1657 run and job states were tied to exact head `21f769d0f80f4651592dac9d80d7a7ee3dc8a049`.
- #1249 and #1659 states were checked against their unchanged exact heads.
- No code, shared capability page, or foreign-owned PR branch was edited.

## 9. What Did Not Go Smoothly

The lane census still reports many active lanes and an old Cursor handover. The exact crosswalk path had no missing competing-ref changes, and a narrow lease was granted for this documentation update.

## 10. Known Residuals

- #1249 still has four active Julia shards.
- #1657's newest run has four active jobs, including all operating-system checks.
- #1659 needs a fresh base and a successful Windows check.
- Broader numerical parity and uncertainty validation remain outside this CI refresh.

## 11. Team Learning

Refresh only when a run changes or the agreed interval has elapsed, and bind the update to the exact head SHA and run ID. Hardware availability is not a reason to duplicate active CI work.

## 12. Cross-Product Coverage

Covers: current status for three PRs in drmTMB and DRModels.jl.

Does NOT cover: numerical parity, recovery, interval calibration, owner confirmation, merge, release, publication, or deployment.

Editorial check: `slop_check.py` reported 0 findings for this exact report, 2026-11-02. Science: not applicable. Facts: limited to returned GitHub diagnostics. References: no external sources used.
