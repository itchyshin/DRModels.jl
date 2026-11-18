## 1. Goal

Refresh the DRM R–Julia parity crosswalk from current main, merged PR, exact-head CI, and owner-gate evidence without changing source or PR state.

## 2. Implemented

- Added a dated current-main and exact-head refresh to `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`.
- Added a per-slice check-log row and a NEWS fragment for this documentation update.
- Recorded current main SHAs, merged PRs #1247/#1249 and #1659/#1663/#1667, and selected open PR head/base/check states.
- Preserved the distinction between a passing old-head workflow, a current-base check, a preview result, and owner authorization.

## 3a. Decisions and Rejected Alternatives

- Kept the work read-only toward GitHub. No PR was edited, rebased, merged, closed, or commented on.
- Did not start Totoro or DRAC jobs because this refresh queried hosted checks only; no fit, simulation, or local package test was needed.
- Treated #1237 as pending its owner's stated two-version and current-base conditions, despite its recorded four-shard green run.
- Assumption: the crosswalk is the right record for this selected status refresh. If the owner wants a complete census of every open PR, a separate bounded inventory is needed.

## 4. Files Touched

- `docs/dev-log/evidence/2026-11-02-pr-review-crosswalk/README.md`
- `docs/dev-log/after-task/2026-11-04-parity-crosswalk-refresh.md`
- `docs/dev-log/check-log.d/2026-11-04-parity-crosswalk-refresh.md`
- `news/parity-crosswalk-refresh.md`

The pre-existing `tools/__pycache__/runtests_files.cpython-313.pyc` modification was not touched.

## 5. Checks Run

- `gh api repos/itchyshin/DRModels.jl/commits/main` and `gh api repos/itchyshin/drmTMB/commits/main`: recorded the exact current main SHAs and commit times.
- `gh pr view` for the selected open and merged PRs: recorded exact head/base IDs, merge states, merged timestamps, and check rollups.
- `gh run view 41249874224 --repo itchyshin/DRModels.jl --json jobs`: confirmed the four shard names and results for #1237.
- Refreshed drmTMB #1671 at 13:02 UTC: all four Ubuntu shards, source-tree tests, and OS matrix passed on head `fc342a31f4654e95ac2434ed999fb043897ae0b1` with base `8d30ed99a1d6c621dc23ce7fbb067ce7dc4cdcee`. GitHub still reports merge state `UNKNOWN`, no review decision, and no review/comments.
- `gh run list --repo itchyshin/drmTMB --workflow R-CMD-check --branch main`: latest listed run is failed run `40375071745` on old head `02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5`; no run is listed on current main `8d30ed99a1d6c621dc23ce7fbb067ce7dc4cdcee`.
- GitHub compare API: this branch is 43 commits ahead and 55 behind current DRModels.jl main; no PR exists for the branch.
- `git diff --check`: passed after the crosswalk update.
- `python3 ~/shinichi-brain/tools/slop_check.py <absolute crosswalk path>`: completed with 0 findings; 17 pattern hits were reviewed as non-blocking.
- `Rscript ~/shinichi-brain/tools/check-after-task.R <absolute report path>` from this DRM.jl worktree: passed the structure check.
- `python3 ~/shinichi-brain/tools/closeout.py check <absolute report path>`: not passed. The compiler resolved its root to the Shinichi brain repository and encountered 11 unrelated acceptance ledgers without a runnable Node gate. The project-root structure check above is the relevant result; global evidence status remains unknown.
- No package tests, fits, simulations, or builds were run.

## 6. Tests of the Tests

Not applicable. No code or tests changed. The CI facts in the crosswalk are linked to the exact run IDs and head SHAs so a later reviewer can re-open the same evidence.

## 7a. Issue Ledger

- #1237: four shards passed on its recorded head, but the recorded workflow does not establish the owner's Julia 1.10 plus Julia 1 condition, and its base is behind current main.
- #1248 and #1214: exact-head Julia shards failed; keep blocked.
- #1215: Julia checks and docs build passed, but docs preview was cancelled; draft, stale base, and docs overlap remain.
- #1228: Julia checks and docs build passed, preview was cancelled, and GitHub reports `DIRTY`; draft and conflict remain.
- #1671: exact-current-base PR; all CI checks passed on its exact head. GitHub reports merge state `UNKNOWN`, with no review decision; no merge authorization was given for this PR.
- Other selected R PRs without check rollups remain `not verified`; no check rollup was not interpreted as pass or failure.

## 8. Consistency Audit

- Checked the current GitHub main SHAs against the selected PR base SHAs.
- Checked merged PR records against the resulting repository main commits.
- Kept historical refreshes in the README intact and labeled the new section as a dated update.
- Did not infer lane ownership from PR author or branch names. No messages were sent.

## 9. What Did Not Go Smoothly

- The first closeout-template invocation resolved a relative path to the brain repository. That generated template was removed, and the report was recreated at the explicit DRM.jl worktree path. No brain file remains changed by this step.
- The full lane census still shows many active foreign lanes. The scoped README lease was granted; no unrelated file was edited.

## 10. Known Residuals

- This is a selected PR refresh, not a census of all open PRs.
- PR states can change after this snapshot. Refresh exact heads and checks before any PR action.
- The current-main R-CMD-check state remains separate from these PR-specific checks. The earlier Totoro check was stopped and did not complete.
- The crosswalk branch is 43 commits ahead and 55 behind current main and has no PR. Do not push this branch as the two-file status slice; port the documented files to a clean current-main branch after keeping this worktree intact.
- No merge, release, deployment, or publication readiness is established.

## 11. Team Learning

Memory receipt: searched the Shinichi brain for DRM parity, lane ownership, and efficiency guidance; read the after-task and writing-naturalness protocols. `route.py` found no LOAD-FIRST manifest for this worktree. The project `AGENTS.md`, lane preflight, and Graft query shaped the path-scoped review.

Golden Set: not in scope; no source or model behavior changed.

## 12. Cross-Product Coverage

Covers: selected DRM R and Julia PR metadata, exact-head hosted CI state, merged commit identities, and the specific owner gate recorded on #1237.

Does NOT cover: code correctness beyond the linked CI runs; a fresh full R or Julia package check; simulation recovery or interval calibration; all open PRs; unresolved lane ownership; CMR package readiness; merging, release, deployment, or publication.

### Post-draft assessment

Style: 2/10, moderate confidence; the report is specific and evidence-led, with some unavoidable status-table phrasing. Genre/coverage: closeout report, all sections reviewed. Evidence/repair: no located passage needed repair after checking dates, SHAs, and run IDs against GitHub. Gates: science = pass for the narrow status summary; facts = pass for checked claims, with stated snapshot limits; references = pass, all cited PR/run links are direct GitHub records. Provenance: self-review, draft `2026-11-04-parity-crosswalk-refresh.md`, 2026-11-04; prior examples in the crosswalk were seen.

### Addendum, 2026-11-04 post-merge refresh

Fresh GitHub reads show drmTMB #1671 merged at 13:02:19 UTC as `9e2c03f6fdcb9091418cb629f2ca7f30da695e3d`; its exact-head OS matrix, source-tree tests, and four Ubuntu shards passed before merge. The latest R-CMD-check run remains failed on the old October 26 SHA `02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5`; no run for the current R `main` SHA was listed.

DRModels.jl `main` remains `f4fa3aed451636b6ffa442ce2632515513d23b1f`. PR #1237's latest relevant run, `41401878711`, failed Julia 1.10 shards 1/4 and 2/4, while all Julia 1 shards passed. PR #1222 remains draft on an old base with previously green checks only. Selected R PRs #1656, #1657, and #1664 are open with `DIRTY` merge state and no current checks. These candidates remain with their existing owners.

Local checks rerun after this addendum: `git diff --check`, the project-root after-task structure check, and naturalness checks for both edited files passed. The evidence promotion gate passed with the brain as root (8 registered claims). An initial invocation with the package worktree as root failed because that checkout has no `memory/evidence-registry.tsv`; it made no changes. No package tests or model computations were run. The unrelated pycache modification remains untouched. This remains a local, uncommitted checkpoint; the branch is still 43 commits ahead and 55 behind current Julia `main`, so do not push it.

Post-draft assessment for the addendum: 2/10, moderate confidence; wording remains a concise evidence record. Science, factual and reference gates pass for the stated GitHub snapshot. The selected list is not a full PR census. `slop_check.py` reported zero findings on the updated report and crosswalk.

### Addendum, 2026-11-04 13:43 UTC exact-head and owner-decision refresh

Fresh GitHub reads confirm DRModels.jl `main` at
`f4fa3aed451636b6ffa442ce2632515513d23b1f` and drmTMB `main` at
`9e2c03f6fdcb9091418cb629f2ca7f30da695e3d`. The selected exact-head states
are recorded in the crosswalk README.

The main correction is DRModels.jl issue #985: Shinichi's Oct 25 decision
already resolves the docs policy. It directs the owning lane to document the
remaining engine-internal exports, add StatsModels to the docs environment,
and set `checkdocs = :exports`, without un-exporting symbols. PR #1222's older
comment calling that decision open is superseded. Its current diff does not
touch `docs/make.jl` or `docs/Project.toml`, and its base remains stale. Its
recorded Julia and Documenter checks passed on its head, but the PR still
needs an owner-led update and exact-current-base checks before it can close.

Other fresh checks: #1237's latest run `41401878711` failed Julia 1.10 shards
1 and 2; #1248 run `41511040751` failed shards 2 through 4; #1240 remains
failed on shard 2. #1243's recorded Julia and docs checks passed but it remains
a draft on a stale base. drmTMB #1656, #1657, and #1664 remain `DIRTY` with no
current check rollup. No source branch, PR, issue, run, or remote ref changed.

The crosswalk paths had a granted scoped lane lease before this edit. The
crosswalk remains a local uncommitted checkpoint; its branch is 43 commits
ahead and 55 behind current Julia `main`, so this does not authorize pushing
that branch. All other lane work remains preserved.

### Exact-main review branch preparation, 2026-11-04

Created isolated branch `codex/pr-crosswalk-refresh-20261104` from exact
current Julia `main` `f4fa3aed451636b6ffa442ce2632515513d23b1f`, tracked by
issue #1251. Copied the reviewed crosswalk and its report byte-for-byte from
the preserved divergent worktree. Added the required check-log row and NEWS
fragment. The original worktree, its unrelated pycache change, and all other
lanes remain untouched. The issue tracks only this evidence snapshot; it does
not close the broader parity programme or change CMR's deferred
interval-calibration work.
