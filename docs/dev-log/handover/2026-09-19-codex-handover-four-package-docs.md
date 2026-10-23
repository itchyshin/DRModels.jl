# Session Handoff: four-package reader documentation programme

Meta: 2026-09-19 · authored by Codex · destination: a fresh Codex task.

## Critical Context

The D-269 package rename and its public closeout are complete.  This handover
does **not** reopen the rename, change modelling APIs, publish a registry
release, or rename local checkout folders.  The next programme is a
documentation-only, reader-first consistency sweep across `drmTMB`,
`DRModels.jl`, `gllvmTMB`, and `GLLVModels.jl`.

At handoff, a foreign Claude performance lane remains live in this DRM.jl
repository (`claude/lane-speed6-20260919`), with an active lease on `bench/`,
`src/sparse_aug_plsm.jl`, `src/fit_q4_sparse_tmb.jl`, and `src/`.  It is out
of scope for the documentation programme.  Do not touch its paths, reset its
work, or describe it as completed.

## Goal

Make the four package sites consistent and reader-first without erasing useful
technical provenance.  A researcher should be able to answer: **“What kind
of scientific model are you trying to fit?”**, select an appropriate route,
see the supported scope in plain language, and run a first example.

The intended routes are:

| Twin pair | Reader routes |
|---|---|
| drmTMB + DRModels | General distributional models · Phylogenetic comparative models · Meta-analytic models |
| gllvmTMB + GLLVModels | General latent-variable models · Phylogenetic comparative models · Species-distribution/community models |

The navigation architecture and tutorial conventions should match, but the
third label must remain scientific rather than mechanically identical.

## What Was Accomplished Before This Handoff

- The `DRM.jl` → `DRModels.jl` and `GLLVM.jl` → `GLLVModels.jl` Julia rename
  work was completed through unmerged/then reviewed rename PRs and the
  coordinated public closeout.  The renamed GitHub repositories and Pages
  sites are live; no registry publication was made.
- Fresh direct Julia and R-to-Julia smoke tests were recorded for both twins.
  `drmSEM` was checked only as ordinary R behaviour; it is not a Julia-parity
  claim.
- The stats-hours reader-path PR was merged after its portable rendering-hook
  repair and verified Pages deployment.  It is not part of the four package
  source sweep.
- A future-programme note records the route taxonomy and the reader/internal
  boundary at `/Users/z3437171/.codex/memories/extensions/ad_hoc/notes/2026-09-19-tutorial-taxonomy-and-reader-boundary.md`.

## Current Working State

- **Working:** the completed rename/publication closeout; no open source edit
  is owed from that arc.
- **In progress elsewhere:** DRM.jl performance lane `claude/lane-speed6-20260919`.
- **Not yet started:** the four-package article and documentation audit.
- **Protected open work:** do not alter GLLVModels draft PRs #399, #409, #410,
  or #411; retain their stated gates.  Existing open PRs in all four
  repositories are independent until rechecked.

## Key Decisions and Rationale

1. **Reader/internal boundary.** Rendered reader-facing pages and tutorials
   must not show PR or issue numbers, sprint or arc IDs, internal decision
   labels, lane names, or agent references.  Dev logs, handovers, check logs,
   planning notes, and contributor/developer material may retain provenance.
2. **Audit before rewrite.** Classify every candidate occurrence as: (a)
   reader-visible leakage to remove/rewrite, (b) useful developer provenance
   to relocate behind a contributor boundary, or (c) historical record to
   preserve but exclude from rendered reader navigation.
3. **Capability before claims.** Inventory actual supported meta-analytic and
   species/community-model examples before writing landing pages.  Do not
   broaden scientific claims merely to fill a navigation slot.
4. **No API migration.** Package/module spelling is `DRModels` / `GLLVModels`;
   familiar modelling functions such as `drm` and `gllvm` remain modelling API
   concepts, not rename targets.
5. **No release activity.** Do not register, tag, publish, or make a CRAN
   release in this programme.

## Mission-Control Summary

| Area | State | What is covered | Next action by leverage |
|---|---|---|---|
| DRModels / drmTMB rename | complete | package names, public sites, optional bridge compatibility | preserve; audit only reader-facing transition wording |
| GLLVModels / gllvmTMB rename | complete | package names and public site | preserve protected drafts; audit reader pages only |
| Four-package documentation | not started | route taxonomy and hard boundary decided | inventory visible leaks and supported examples |
| DRM.jl performance | foreign live lane | speed work under an active lease | leave untouched; recheck at next handoff |

## Landing State

| Artifact / branch | Committed | Pushed | PR | State |
|---|---:|---:|---|---|
| `handover/2026-09-19-codex-four-package-docs` @ `29257efce` (this handover) | yes | yes | [#777](https://github.com/itchyshin/DRModels.jl/pull/777) open | CARRIED-OVER pending human review; do not auto-merge |
| `claude/lane-speed6-20260919` | yes, unpushed | no | none observed locally | CARRIED-OVER; foreign live performance lane |

FINDINGS-OF-RECORD: none.  This handover records a user-approved programme
shape, not a new scientific finding.

## Next Immediate Steps

1. Start a fresh Codex task and run `ultra-plan` with this document as the
   programme brief.  Do not begin broad edits before its acceptance ledger
   names the rendered-site boundary, the four repositories, and every human
   gate.
2. Run `lane_preflight.sh` separately in `drmTMB`, `DRModels.jl`, `gllvmTMB`,
   and `GLLVModels.jl`; read each current handover, coordination board, and
   open-PR state.  Respect protected/foreign lanes.
3. Build four read-only inventories: rendered navigation/tutorial sources,
   reader-visible internal-reference leaks, existing runnable examples, and
   current route-capability evidence.  Keep the results as a reviewable
   manifest before editing pages.
4. Present the inventory and a package-by-package implementation plan for
   approval.  Gate any broad rendered-site deployment and every merge in the
   ordinary repository-specific way.
5. Execute only approved, isolated documentation slices; run their native
   rendering/checks and inspect served pages before describing a route as
   complete.

## Blockers and Open Questions

- Exact meta-analytic scope in drmTMB/DRModels must be measured from current
  examples and tests before public wording is written.
- Exact species-distribution/community scope in gllvmTMB/GLLVModels likewise
  needs a current inventory rather than inherited labels.
- The foreign DRM.jl speed lane must be reconciled before any `src/` or
  benchmark work, but it does not block a docs-only programme in separate
  worktrees.

## Gotchas and Failed Approaches

- Do not use a blind global replacement for internal identifiers: historical
  records and developer provenance are deliberately retained.
- Do not infer a repository-wide state from one working tree; recheck each
  repository's `origin/main`, open PRs, and current Pages output.
- Do not make CI the compute engine for performance/recovery evidence.  Use
  local/Totoro/DRAC according to the compute rules when a later evidence task
  truly needs it.

## Files Created / Modified

- `docs/dev-log/handover/2026-09-19-codex-handover-four-package-docs.md` —
  this handover; no package source, rendered page, or API file changed.

## How to Resume

From the repository root, start a fresh Codex task and paste:

```text
Rehydrate from docs/dev-log/handover/2026-09-19-codex-handover-four-package-docs.md and AGENTS.md. Reconcile every repository's live lane state, then run ultra-plan for the four-package reader documentation programme. Execute only the OWED Next Immediate Steps.
```

Codex owns live rendering, package checks, and any validated example runs.
Planning, prose, and audits may be delegated, but no agent may edit protected
or foreign lanes without a fresh lease and explicit scope check.
