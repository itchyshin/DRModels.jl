# After Task: Reader PR triage

## 1. Goal

Classify the stacked reader-documentation PRs in DRModels.jl (the 7 the
2026-09-21 Claude handover names, plus any other open PR touching `docs/src/`
or `docs/make.jl`) into OWED, DONE, RETRACTED, or PROTECTED, so the maintainer
can decide ownership and merge order. No prose, no merges, no test or
Documenter runs in this slice.

## 2. Implemented

- Wrote `docs/dev-log/reader-pr-triage-2026-09-21.md`: a 10-row table (#800,
  #798, #797, #796, #795, #789, #779 named by the handover, plus #781, #770,
  #576 found by scanning every open PR's file list for `docs/src/` or
  `docs/make.jl` touches), a #800 merge-readiness verdict against four named
  conditions, a conflict-probe log, a "not done here" section, and a
  "recommended next" section.
- Revised that table after an independent read-only review (Fable: Rose +
  Pat, S4) found 0 blocking items and 8 should-fix / 5 nit items: rescored the
  `scaling-sweep` CI skip as structural rather than failed, downgraded a
  `ci-ok`/`docs` mismatch from an alarm to a by-design note, extended the
  "confirm ownership" question from #800 alone to #800/#798/#796, added a
  head-SHA column to every row, replaced one row's speculation with a
  workflow citation, softened an unnamed-owner claim on #770, flagged #789's
  ownership as provisional, and noted that no CI log was read.
- Appended two dated sections to PR #801's body: a classification summary
  and, later, a one-line note recording the revision.

## 3a. Decisions and Rejected Alternatives

- Withheld the merge on #800 even though three of its four readiness
  conditions held outright, because the task's own instructions require the
  maintainer's confirmation of a paused Codex automation first. Merging on
  the strength of three-of-four was rejected as exceeding the task's
  authorization.
- Used `git merge-tree --write-tree` (writes a tree object, creates no branch
  and no worktree) to probe for conflicts and count conflicted files, instead
  of checking out and rebasing each PR's branch. Rebasing was rejected: the
  task forbids touching PR branches, and a probe needed to be cheap and
  repeatable across ten branches without side effects.
- Kept #781, #770, and #576 in the table as PROTECTED rather than silently
  dropping them for being outside the handover's named list. Dropping them
  was rejected because they hold files (`docs/src/rosetta.md`, `docs/make.jl`,
  `test/runtests.jl`) that the OWED PRs will eventually collide with, and the
  maintainer needs that visible even though this lane cannot act on it.
- Declined to diagnose or weaken #795's failing Documenter check inside this
  triage. The PR's own stated purpose ("retain valid protection") suggests
  the failure could be the guard correctly firing on a real gap; reading the
  actual CI log and deciding that question was judged out of scope for a
  classification-only task.

## 4. Files Touched

- `docs/dev-log/reader-pr-triage-2026-09-21.md` (created, then revised)
- `docs/dev-log/after-task/2026-09-21-reader-pr-triage.md` (this file)
- `docs/dev-log/check-log.d/2026-09-21-reader-pr-triage.md`

## 5. Checks Run

- `gh pr list -R itchyshin/DRModels.jl --state open --json number,title,headRefName,isDraft,mergeable,files`:
  returned 13 open PRs; used to find PRs beyond the handover's 7 that touch
  `docs/src/` or `docs/make.jl` (added #781, #770, #576).
- `gh pr view <n> -R itchyshin/DRModels.jl --json number,title,headRefName,isDraft,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup,files,updatedAt,author`
  for all 10 PRs: recorded draft state, mergeable/mergeStateStatus, every
  check's name and conclusion, file list, and author.
- `git merge-base --is-ancestor origin/<branch> origin/main` for all 10
  branches: exit false (not an ancestor) in every case, so no PR is DONE.
- `git diff --stat origin/main...origin/<branch>` for all 10 branches:
  non-empty in every case.
- `git merge-tree --write-tree origin/main origin/<branch>` for the 4
  CONFLICTING PRs and the 3 PRs with failing checks (7 probes total): exit 1
  with 3, 3, 1, and 1 conflicted paths for #779, #781, #770, and #576
  respectively (paths listed in the table); exit 0, no conflicts, for #797,
  #795, and #789.
- `git diff --check` before each commit on this branch: clean every time,
  across three commits (the original triage, the S4 revision, this
  after-task/check-log pair).
- Refs snapshot diff (`git for-each-ref` before/after each write): only
  `refs/heads/claude/drmodels-reader-arc-handover-20260921` and its
  remote-tracking counterpart moved, except once when
  `refs/remotes/origin/gh-pages` also moved, a side effect of `git fetch
  origin` (a remote-tracking ref updated by fetch, not a write by this
  session).
- Direct reads of `.github/workflows/CI.yml` and
  `.github/workflows/Documenter.yml` during the revision pass, to verify the
  S4 review's citations before applying them: `CI.yml:131-134`
  (`scaling-sweep`'s `if: github.event_name == 'workflow_dispatch'`),
  `CI.yml:119-122` (`ci-ok`'s `needs: [test]`, `if: always()`), and
  `Documenter.yml:70-71` and `:133` (the audit scripts already run inside
  `docs`, and the required contexts are exactly `["docs", "ci-ok"]`). All
  four reproduced exactly as quoted.
- `python3 ~/shinichi-brain/tools/slop_check.py docs/dev-log/after-task/2026-09-21-reader-pr-triage.md`:
  run until it reported PASS on this file before committing.

## 6. Tests of the Tests

Not applicable. This slice produced a PR classification table and two
dev-log records, not code or a testable runtime behaviour, so there is no
test suite whose own correctness needs checking. The nearest equivalent, not
trusting a claim without re-deriving it from a primary source rather than
from a title or a prior summary, is what the Checks Run and Consistency
Audit sections record instead: every classification traces to a `gh` JSON
field, a `git` command's exit code, or a directly read workflow-file line.

## 7a. Issue Ledger

- Fixed (by the S4 revision): a self-contradictory reading of the
  `scaling-sweep` skip that scored it both "FALSE" and "routine" in the same
  document; a red-flag framing of `ci-ok`/`docs` that borrowed an unrelated
  comparison (a different repo's 2026-09-04 finding) for a by-design behaviour
  in this repo; an unnamed-owner claim on #770; an uncited speculation on
  #795.
- Not fixed, out of scope: the actual root cause of the `Julia 1 - shard 2/4`
  failure shared by #797, #789, and #779, and the actual content of the
  failing `docs`/Documenter build on #795. Both need a CI log read, which
  this triage task does not authorize.

## 8. Consistency Audit

Every row's classification rests on at least two independent sources: the
`gh pr view` JSON (draft, mergeable, checks, files) and either a `git
merge-tree` probe or a direct `git diff --stat` / `merge-base` check against
`origin/main`. The three PROTECTED PRs were cross-checked against the
handover's own boundary list (formula grammar, `src/` likelihood code,
parity numbers, engine work); #781 was additionally cross-checked against its
own linked handover
(`docs/dev-log/handover/2026-09-19-claude-handover-speed6-test-hygiene.md`),
which independently confirms it as a closed sub-lane of a separate,
already-named arc. The revision pass re-verified the S4 review's own workflow
citations by reading the named lines directly rather than trusting the
findings file at face value; all reproduced exactly.

## 9. What Did Not Go Smoothly

The first pass scored the `scaling-sweep` CI skip inconsistently: the #800
verdict called it a failed condition ("(a) ... FALSE") in the same document
that called the identical fact "routine" elsewhere, a contradiction a
maintainer would otherwise have to resolve themselves. The fix was to read
the workflow's actual `if:` condition rather than infer the skip's meaning
from its name, and then say the same thing about it everywhere it appears.
Separately, the first two commits on this branch (`a17fdff0d`, `e2aebf643`)
carry a `Co-Authored-By: Claude Opus 5` trailer, because that is what the
task brief spawning that work specified. The coordinator has since said that
instruction was its own error; this session is Claude Sonnet 5, and this
commit uses the correct trailer. The two earlier commits were not rewritten,
since fixing them would need rewriting already-pushed history, which this
task forbids; the mismatch is recorded here instead.

## 10. Known Residuals

- Handover steps 2 through 5 (write the reader-first prose, rebuild the model
  map and diagnostics pages, render and read the public route, run a Rose
  claim/limits pass) remain undone, deliberately: they would edit the same
  pages six of the ten triaged PRs already hold.
- Three decisions still need the maintainer's word, not this lane's: whether
  the paused Codex automation is confirmed (gates #800, #798, #796), whether
  the reader lane may edit `src/` docstrings (gates #789, whose files partly
  overlap PROTECTED #770), and who sequences #770/#576/#779's overlapping
  files (`docs/src/rosetta.md`, `docs/make.jl`) before #779 can rebase
  cleanly.
- The `Julia 1 - shard 2/4` failure (shared by #797 and #789) and the
  `docs`/Documenter failure on #795 are recorded but not diagnosed; no CI log
  was read in either triage pass.

## 11. Team Learning

A checks-summary column that only counts green, failing, and skipped invites
a reader to treat every skip as equivalent; it is not, once a workflow's
`if:` condition shows a skip is structural (cannot run on this event type at
all) rather than incidental (this run happened not to trigger it). Reading
the workflow file itself, not just the check names, is the cheap way to tell
those apart, and doing it here caught a self-contradiction that an
independent review then had to flag. Separately: an attribution trailer is
easy to get wrong across a multi-agent, multi-model programme (Opus versus
Sonnet, across DRModels.jl and its GLLVModels.jl sibling); once a wrong
trailer is pushed, the cheaper fix is to record and correct it going
forward, not to rewrite pushed history for a commit-message line.

Memory receipt: read `AGENTS.md`'s reader-documentation handover boundary
list, `docs/dev-log/coordination-board.md`, and this repo's
`.github/workflows/CI.yml` and `Documenter.yml` directly. No brain search was
needed for this slice; the task was self-contained inside one repo's PR
metadata and workflow files.

Golden Set: not in scope. This slice classified pull requests; it changed no
code, no numerical claim, and no public documentation page.

## 12. Cross-Product Coverage

This slice covers the mergeability, CI state, file overlap, and ownership
status of the 10 open DRModels.jl PRs that touch `docs/src/` or
`docs/make.jl` as of 2026-09-21. It does NOT cover: the content correctness
of any PR's prose, the root cause of any failing CI check, GLLVModels.jl's
parallel triage (tracked separately in its own PR #444), or any
package, API, or engine behaviour.
