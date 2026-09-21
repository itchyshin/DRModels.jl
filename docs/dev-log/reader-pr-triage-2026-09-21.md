# Reader PR triage, 2026-09-21

Branch: `claude/drmodels-reader-arc-handover-20260921` (PR #801, draft). Baseline: `main = 2050350d4`. Read captured 2026-09-21T17:18:53Z (UTC). Author: Claude, S2 triage lane for the four-package reader-documentation programme (see `docs/dev-log/handover/2026-09-21-claude-handover.md`).

Scope: the 7 PRs named in the handover's "Current reader PRs" table, plus every other open PR whose files touch `docs/src/` or `docs/make.jl` (found by scanning all open PRs' file lists). That added three PRs (#781, #770, #576) that are not part of the reader programme.

## Table

| PR | Title | Branch | Draft | Mergeable | Checks (green/failing/pending/skipped) | Files (count; reader-facing) | Overlaps with | Classification | Recommended action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #800 | docs: make first DRModels route reader-first | `codex/getting-started-reader-20260921` | no | MERGEABLE (CLEAN) | 13/0/0/1 (skipped: scaling-sweep) | 1; `docs/src/getting-started.md` | none | OWED | Merge withheld pending maintainer confirmation (see verdict below). Once confirmed: merge as is. |
| #798 | docs: define distributional regression before package language | `codex/landing-definition-20260921` | yes | MERGEABLE (CLEAN) | 13/0/0/1 (skipped: scaling-sweep) | 1; `docs/src/index.md` | none | OWED | Undraft, quick content review against #800's route, merge. |
| #797 | docs: route relmat readers to public guidance | `codex/relmat-reader-20260921` | yes | MERGEABLE (BLOCKED) | 11/2/0/1 (failing: `Julia 1 - shard 2/4`, `ci-ok`; skipped: scaling-sweep) | 1; `docs/src/tutorials/relmat-known-matrices.md` | none | OWED (blocked) | Diagnose the shard 2/4 failure before touching content. Handover: "diagnose, never force." No git conflict (merge-tree exit 0, no conflicted paths). |
| #796 | docs: route structured effects through overview | `codex/model-map-structural-route-20260920` | yes | MERGEABLE (CLEAN) | 13/0/0/1 (skipped: scaling-sweep) | 1; `docs/src/model-guides/model-map.md` | none | OWED | Undraft, review, merge. |
| #795 | test: guard documentation entry routes and next steps | `codex/reader-flow-gate-20260921` | yes | MERGEABLE (BLOCKED) | 10/1/0/2 (failing: `docs`/Documenter build; skipped: scaling-sweep, docs-preview) | 2; none are `docs/src/` pages (`tools/reader_surface_audit.py`, `tools/tests/test_reader_surface_audit.py`) | none | OWED (blocked) | Do not weaken the guard to force it green. Handover: "retain valid protection" - the failing Documenter build may be this PR correctly flagging a real gap in the current reader routes. Investigate before any other reader PR merges, since it could re-fail differently once #800/#798/#796 land. No git conflict (merge-tree exit 0). |
| #789 | docs: remove process history from public reference docstrings | `codex/public-reference-docstrings-20260920` | yes | MERGEABLE (BLOCKED) | 11/2/0/1 (failing: `Julia 1 - shard 2/4`, `ci-ok`; skipped: scaling-sweep) | 15; all `src/*.jl` docstrings (reader-facing only once rendered via Documenter `@autodocs`, no standalone `docs/src/` page) | #770 (`src/beta.jl`, `src/gamma.jl`, `src/negbinomial.jl`, same files, not yet conflicting per merge-tree) | OWED (blocked) | Diagnose the shard 2/4 failure. Sequence after #770 if #770 lands first (same three files); re-probe for conflicts at that point. No git conflict against `main` today (merge-tree exit 0). |
| #779 | docs: make R migration guidance reader-facing | `codex/drmodels-migration-navigation-20260919` | no | CONFLICTING (DIRTY) | 11/2/0/1 (failing: `Julia 1 - shard 2/4`, `ci-ok`; skipped: scaling-sweep) | 4; `docs/make.jl`, `docs/src/reference/deprecated-marker-internals.md`, `docs/src/rosetta.md`, `tools/tests/test_docs_navigation.jl` | #770 (`docs/src/rosetta.md`), #576 (`docs/make.jl`) | OWED (needs rebase) | Handover: "dirty; do not force merge." Rebase onto current `main` (3 conflicted paths, see Conflict probes) once #770/#576 are resolved by their own lanes, then re-review; do not rebase in this triage run. |
| #781 | speed6: q=4 phylo ML speed arc (S3/S5b/S5c/S5d) + test-suite hygiene | `claude/lane-speed6-20260919` | no | CONFLICTING (DIRTY) | 13/0/0/1 (skipped: scaling-sweep) | 31; only `docs/src/reference/engine-internals.md` is reader-facing, rest is `src/`, `test/`, `bench/`, `LOOP/`, `report/` performance/engine work | #770 (`test/runtests.jl`) | PROTECTED | Not in the handover's 7-PR list and not reader-documentation work. It is the separate "speed6" performance/test-hygiene arc (own handover: `docs/dev-log/handover/2026-09-19-claude-handover-speed6-test-hygiene.md`, "orchestrator session shinichi-82"). Its one `docs/src/` touch (`engine-internals.md`) is why the sweep caught it. AGENTS.md/handover boundary: "Protected: ... src/ likelihood code ... engine work." Hand back to the speed6 lane; do not merge or rebase here. Also incidentally overlaps (out of table scope) with open PR #793 on `src/gaussian_bivariate.jl`. |
| #770 | fix(scales): sigma(fit) reports the scale the likelihood scored, on all 12 NB2/Gamma/Beta routes (audit M2, D-268) | `fix/report-clamped-sigma-20260915` | no | CONFLICTING (DIRTY) | 13/0/0/1 (skipped: scaling-sweep) | 9; only `docs/src/rosetta.md` is reader-facing, rest is `NEWS.md`, `src/beta.jl`/`gamma.jl`/`negbinomial.jl`, `test/` | #779 (`docs/src/rosetta.md`), #789 (`src/beta.jl`, `src/gamma.jl`, `src/negbinomial.jl`), #781 (`test/runtests.jl`) | PROTECTED | Not in the handover's 7-PR list. A likelihood/scale correctness fix ("src/ likelihood code" per the boundary list), oldest is 2026-09-15. Its `docs/src/rosetta.md` touch is incidental (vocabulary update for the new behaviour). Owned by whatever lane opened the M2/D-268 audit; out of scope here. |
| #576 | docs: R<->Julia parity scoreboard (GLLVM.jl-style catch-up page) | `docs/drmtmb-parity-scoreboard` | yes | CONFLICTING (DIRTY) | 13/0/0/1 (skipped: scaling-sweep) | 2; `docs/make.jl`, `docs/src/drmtmb-parity.md` (new page) | #779 (`docs/make.jl`) | PROTECTED | Not in the handover's 7-PR list. A parity-scoreboard page; the handover boundary names "parity numbers" as Protected. Oldest PR found (updated 2026-09-06); out of scope here. |

No PR in this set is DONE: `git merge-base --is-ancestor origin/<branch> origin/main` returned false for all ten, and `git diff --stat origin/main...origin/<branch>` was non-empty for all ten.

No PR in this set is RETRACTED: nothing in the handover, `AGENTS.md`, `HANDOVER.md`, `ROADMAP.md`, or `docs/dev-log/coordination-board.md` withdraws any of the seven reader PRs.

Every `statusCheckRollup` entry read `status: COMPLETED`; there were no pending checks on any of the ten PRs at read time.

🔴 Worth a maintainer's attention, found in passing: on #795 the aggregate `ci-ok` check is SUCCESS even though the `docs` (Documenter build) check on the same commit is FAILURE. `ci-ok` does not appear to gate on the Documenter build. This is the same shape of problem flagged for gllvmTMB on 2026-09-04 ("a CI green whose check step was skipped by the fast-pass path"): a green aggregate check that does not mean what it looks like it means. Not investigated further here; flagged for whoever owns `ci-ok`'s definition in this repo's workflow.

## #800 verdict (step 5, no merge performed)

Checked at `main = 2050350d4`, 2026-09-21T17:18:53Z (UTC):

- (a) every CI check green, none pending or skipped: **FALSE**. `scaling-sweep` is SKIPPED (13 green / 0 failing / 0 pending / 1 skipped).
- (b) not a draft: **TRUE**. `isDraft: false`.
- (c) mergeable MERGEABLE: **TRUE**. `mergeable: MERGEABLE`, `mergeStateStatus: CLEAN`.
- (d) the handover names it as this lane's to land: **partially**. Quoting the handover's table: "| [#800](https://github.com/itchyshin/DRModels.jl/pull/800) | first getting-started route | green/clean; confirm ownership before merge |". It calls the PR green/clean, but the instruction is conditioned on an ownership confirmation that has not happened in this triage run.

Not all four hold cleanly (a fails on the letter of the check, and d is conditional, not unconditional). **Merge withheld pending maintainer confirmation** in every case, per this task's instructions: this run does not merge #800 even if all four had held, because the maintainer is confirming a paused Codex automation first.

## Conflict probes

`git merge-tree --write-tree origin/main origin/<branch>`, trimmed to the conflict summary lines (exit 1 = conflicts found; blank "exit 0" entries below are the CI-failing-but-git-mergeable PRs, probed per instruction even though `gh` already reported them MERGEABLE):

```
#779 codex/drmodels-migration-navigation-20260919 vs main: exit 1, 3 conflicted paths
  CONFLICT (content): Merge conflict in docs/make.jl
  CONFLICT (content): Merge conflict in docs/src/rosetta.md
  CONFLICT (content): Merge conflict in tools/tests/test_docs_navigation.jl

#781 claude/lane-speed6-20260919 vs main: exit 1, 3 conflicted paths
  CONFLICT (content): Merge conflict in test/test_joint_missing_bridge.jl
  CONFLICT (content): Merge conflict in test/test_joint_missing_two_predictor.jl
  CONFLICT (content): Merge conflict in test/test_joint_missing_uncertainty.jl

#770 fix/report-clamped-sigma-20260915 vs main: exit 1, 1 conflicted path
  CONFLICT (content): Merge conflict in NEWS.md
  (docs/src/rosetta.md, src/beta.jl, src/gamma.jl, test/runtests.jl all auto-merged clean against main)

#576 docs/drmtmb-parity-scoreboard vs main: exit 1, 1 conflicted path
  CONFLICT (content): Merge conflict in docs/make.jl

#797 codex/relmat-reader-20260921 vs main: exit 0, no conflicts
#795 codex/reader-flow-gate-20260921 vs main: exit 0, no conflicts
#789 codex/public-reference-docstrings-20260920 vs main: exit 0, no conflicts
```

`#800`, `#798`, `#796` were not probed: `gh` already reports `mergeable: MERGEABLE` and all their checks are green (aside from the expected `scaling-sweep` skip), so neither trigger condition (CONFLICTING or failing checks) applies.

## Not done here

Handover steps 2 through 5 (write the distributional-regression-first landing prose, rebuild the model map and diagnostics pages for a reader's seat, sweep neighbouring pages, render and read the public route, run a Rose claim/limits pass) are deferred until the PR set above is settled. Six of the seven reader PRs (#798, #797, #796, #795, #789, #779) each hold a distinct piece of the same small set of pages (`index.md`, `getting-started.md`, `model-map.md`, `relmat-known-matrices.md`, `rosetta.md`, `deprecated-marker-internals.md`, plus the reference docstrings and the reader-flow guard test) that any new prose pass would also need to touch. Writing new prose on top of open, unreconciled PRs on the same files would either collide with them or silently duplicate their content once they land. Steps 2 through 5 should start only after the maintainer has acted on the "Recommended next" ownership calls below, at which point the actually-current state of each page can be re-read from `main`.

## Recommended next

1. **Land first, in order, once each is undrafted and briefly reviewed:** #800 (pending the maintainer's Codex-automation confirmation per the verdict above), then #798, then #796. All three are clean, green apart from the routine `scaling-sweep` skip, and touch no file any other open PR touches.
2. **Diagnose before merging:** #797 and #789 both fail the same two checks (`Julia 1 - shard 2/4`, `ci-ok`) with no git conflict; #795 fails `docs`/Documenter with no git conflict, and its own PR's stated purpose ("retain valid protection") means the failure may be intentional signal rather than a bug to silence. Investigate the shard 2/4 failure once (it repeats across #797 and #789, so it may share a cause) before deciding whether either can merge as is.
3. **Rebase, then review, not now:** #779 needs a rebase onto `main` to clear 3 conflicted paths. Because two of those paths (`docs/src/rosetta.md`, `docs/make.jl`) are also held by PROTECTED PRs #770 and #576, rebasing #779 now would likely need re-rebasing after #770/#576 move. Wait for those two lanes, or ask the maintainer to sequence explicitly.
4. **Hand back, do not merge from this lane:** #781 (speed6 performance arc), #770 (scale/likelihood audit fix), #576 (parity scoreboard). None are reader-documentation work; each is PROTECTED content per the handover's boundary list. Their file overlaps with #779, #789, and each other (see table) are relevant context for whoever does sequence the merges, but resolving those overlaps is not this lane's call.
