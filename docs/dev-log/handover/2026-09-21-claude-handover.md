# Claude Handover — DRModels.jl reader-documentation close

**Date:** 2026-09-21 (America/Edmonton)
**From:** Codex
**To:** Claude Code
**Repository:** `itchyshin/DRModels.jl`

## Critical context

This is the DRModels.jl part of the four-package reader-first documentation
programme. The reader is often a biology, ecology, evolution, or environmental
science PhD student who may be new to Julia. They must be able to answer, in
order: **what is this model for; can I fit my data; what must I check before
trusting it?** Do not add figures, badges, release work, API changes, or engine
work in this arc.

DRModels.jl is a Julia package for distributional regression: predictors can
affect the expected response and also its spread or other distributional
features. It is an independent Julia companion to drmTMB, not an installation
requirement for drmTMB and not a wrapper around its GPL source.

**FINDINGS-OF-RECORD: none.**

## Ten-milestone ledger

| Milestone | Status for this handover |
| --- | --- |
| 1. CI recovery and clean merges | Recheck live checks before every merge. |
| 2. GLLVModels first route | Owned by the GLLVModels companion handover. |
| 3. drmTMB core learning pages | Owned by drmTMB Claude PR #1418. |
| 4. gllvmTMB safety route | Owned by gllvmTMB Claude PR #1317. |
| 5. DRModels beginner routes | **Primary work here:** landing, model map, diagnostics, runnable first fit. |
| 6. GLLVModels follow-on routes | Owned by the GLLVModels companion handover. |
| 7. Cross-site vocabulary | Keep plain definitions consistent across all four sites. |
| 8. Internal-language sweep | Remove public process, record, receipt, and unexplained implementation language. |
| 9. Rendered-reader audit | Build and inspect the actual site and first visible route. |
| 10. Rose close-out | Claims, boundaries, CI, Pages, and only then clean owned merges. |

## Current reader PRs — reconcile, do not duplicate

| PR | Reader contribution | Current instruction |
| --- | --- | --- |
| [#800](https://github.com/itchyshin/DRModels.jl/pull/800) | first getting-started route | green/clean; confirm ownership before merge |
| [#798](https://github.com/itchyshin/DRModels.jl/pull/798) | define distributional regression on landing | draft, green |
| [#797](https://github.com/itchyshin/DRModels.jl/pull/797) | relationship-matrix public guidance | draft, blocked; diagnose, never force |
| [#796](https://github.com/itchyshin/DRModels.jl/pull/796) | structural-effects overview | draft, green |
| [#795](https://github.com/itchyshin/DRModels.jl/pull/795) | entry-route regression guard | draft, blocked; retain valid protection |
| [#789](https://github.com/itchyshin/DRModels.jl/pull/789) | remove process history from reference help | draft, blocked |
| [#779](https://github.com/itchyshin/DRModels.jl/pull/779) | reader-facing R migration guidance | dirty; do not force merge |

The baseline is `main` at `2050350d4` (`docs: clarify uncertainty and
model-fit interpretation (#799)`). It does not imply that the open work is
included.

## Current state and boundaries

- **Working:** this branch contains only this handover note.
- **In progress:** reconcile the listed PRs into a reader order, then land
  reviewable slices rather than a wholesale rewrite.
- **Protected:** formula grammar, `src/` likelihood code, parity numbers,
  version/release/registry work, figures/badges, and optional R-bridge code.
- **Do not stage:** generated site output or files owned by other PRs without
  an explicit transfer.

## OWED next immediate steps

1. Read `AGENTS.md`, `HANDOVER.md`, `ROADMAP.md`, and
   `docs/dev-log/coordination-board.md`; compare all PRs above with live
   GitHub and classify each `OWED`, `DONE`, `RETRACTED`, or `PROTECTED`.
2. Put the scientific question first: explain distributional regression in
   ordinary language, show one small runnable fit, then introduce `bf()`,
   `sigma`, correlation, or structured effects only as needed.
3. Make the model map and diagnostics help a reader choose and check a model,
   not decode internal development history. Sweep neighbouring pages for the
   same mistake (Rose principle).
4. Keep honest limits visible: a successful fit is not scientific validation;
   the R bridge is optional; claims must match documented evidence.
5. Render and read the public route from a novice’s seat. Ask: “what can I fit
   today, and what is the next click?” Then complete a Rose claim/limits pass.
6. Merge only an up-to-date, clean, settled-green PR with explicit ownership.
   Do not merge blocked or dirty siblings merely to shrink the list.

## Verification

Claude may write/review prose; use Codex for live Julia compilation or rendering
if Claude lacks the toolchain.

```sh
tools/lane_preflight.sh .
git status --short --branch
git diff --check
julia --project=docs docs/make.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Inspect the rendered preview as well as source Markdown and record exact
commands/outcomes in the PR.

## Linked handovers

| Repository | Handover / PR | Boundary |
| --- | --- | --- |
| drmTMB | [#1418](https://github.com/itchyshin/drmTMB/pull/1418) | designated Claude repair of intentionally red reader contract |
| gllvmTMB | [#1317](https://github.com/itchyshin/gllvmTMB/pull/1317) | designated Claude first-tutorial rewrite |
| GLLVModels.jl | `claude/gllvmodels-reader-arc-handover-20260921` | GLLVM landing, limits, and follow-on routes |

## Landing state

| Artifact / branch | Committed | Pushed | PR | State |
| --- | --- | --- | --- | --- |
| `claude/drmodels-reader-arc-handover-20260921` | yes | pending | none yet | CARRIED-OVER: push and open a draft PR before a fresh Claude session relies on this note. |

## How to resume

```text
Read AGENTS.md and docs/dev-log/handover/2026-09-21-claude-handover.md. Run the handover rehydration steps, reconcile them with the current git state, then continue only the OWED Next Immediate Steps.
```
