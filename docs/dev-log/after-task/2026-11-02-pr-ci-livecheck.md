## 1. Goal

Refresh exact-head CI evidence for three open R and Julia parity pull requests, record Totoro and DRAC availability, and preserve work owned by other lanes.

## 2. Implemented

- Added a 22:01 UTC diagnostic snapshot to `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`.
- Confirmed #1249 still has four active Julia shards; #1657 has a newer run on its same head; #1659 has cancelled Windows checks and a stale base.
- Recorded that Totoro and DRAC are available. No host computation was needed for this metadata refresh.

## 3a. Decisions and Rejected Alternatives

- Left the shared capability page and all PR branches untouched. This lane owns only its crosswalk notes and this report.
- Did not merge or contact PR owners. None of the three has complete current evidence for a merge decision.
- Did not launch a parity or coverage run. Host availability does not identify a specific run or validation target.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-02-pr-ci-livecheck.md`
- `docs/dev-log/check-log.d/2026-11-02-pr-ci-livecheck.md`

The pre-existing modified Python bytecode file was left untouched.

## 5. Checks Run

- Lane preflight for the crosswalk README found no competing ref changes on that path.
- Exact-head GitHub diagnostics were read for DRModels.jl #1249 and drmTMB #1657 and #1659.
- Git status and branch identity were checked before closeout. `git diff --check` passed.
- `check-after-task.R` passed. `slop_check.py` reported 0 findings across 544 words and no em dashes.
- No package tests, fits, simulations, benchmarks, or model checks were run.

## 6. Tests of the Tests

Not applicable. This slice updates CI status documentation only. The recorded GitHub checks belong to the named PR heads and do not validate this documentation change.

## 7a. Issue Ledger

- No issue or PR metadata was changed.
- #1249, #1657, and #1659 remain open. Their exact heads, checks, and next steps are recorded in the crosswalk.

## 8. Consistency Audit

- PR heads and workflow run IDs were matched to returned diagnostics.
- #1659's base SHA was compared with current drmTMB main and marked stale.
- The record separates point-estimate parity from interval coverage and calibration.
- Other lanes' source files and the main capability-status page were not edited.

## 9. What Did Not Go Smoothly

The first lease-list call stalled before returning. A later exact-path lease was granted, and the documentation update proceeded under that lease.

## 10. Known Residuals

- Four Julia shards remain active for #1249.
- #1657's new source-tree and OS-matrix jobs are not yet complete; its earlier exact-head run still has one Ubuntu shard active.
- #1659 still needs a current-base refresh and a successful Windows result.
- The wider R and Julia parity backlog remains open. Availability of Totoro and DRAC does not itself complete a validation campaign.

## 11. Team Learning

When a file-owner question is unclear, state it plainly: another lane owns the shared capability page, so this lane will only edit its crosswalk notes. Record hardware availability separately from approval and scope for a specific compute run.

## 12. Cross-Product Coverage

Covers: exact-head CI status for three parity PRs and host availability as reported by Shinichi.

Does NOT cover: model execution, R to Julia numerical parity, recovery, interval availability or calibration, PR owner confirmation, merge, release, publication, or deployment.

Editorial check: `slop_check.py` reported 0 findings for this exact report, 2026-11-02. Science: not applicable. Facts: pass against timestamped GitHub diagnostics and Shinichi's host-availability message. References: no external sources used.
