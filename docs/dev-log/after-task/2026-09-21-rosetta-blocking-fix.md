# After Task: Rosetta and figure-gallery blocking fixes

## 1. Goal

Fix the three BLOCKING reader findings against `docs/src/rosetta.md` and
`docs/src/diagnostics-and-validation/figure-gallery.md` named in the DRModels.jl
section of `2026-09-21-cross-site-reader-audit.md`, and nothing else
(maintainer decision recorded in that audit: "yes, one PR, no other scope").

## 2. Implemented

- `rosetta.md:127`: the row said drmTMB's `summary(fit)` has no Julia
  equivalent ("no `summary` method"). `Base.summary(fit::DrmFit)` is defined
  (`src/summary.jl:276`) and `getting-started.md` runs `summary(fit)` in its
  worked example, so the old row told an R user the method they had just seen
  does not exist. Row now reads `summary(fit)` (prints the Wald coefficient
  table).
- `rosetta.md:119`: the row marked `rho12(fit)` "planned (parity gap)".
  `rho12(fit::DrmFit)` is implemented (`src/summary.jl:117`), is named in
  `getting-started.md`'s post-fit accessor list, and `capabilities.md` lists
  the `rho12(fit)` accessor as **Tested**. Row now reads `rho12(fit)`, the
  same bare-call convention this table already uses for every other available
  accessor (`sigma(fit)` / `sigma(fit)`, `family(fit)` / `family(fit)`, ...),
  with no "parity gap" language.
- `figure-gallery.md:7` and `:47-48`: both occurrences of "Florence's house
  contract" (an agent persona name plus an undefined phrase) removed. Line 7
  now names the actual contract in one clause; the fuller description at
  47-48 keeps its existing plain-language sentence (pale compatibility
  region, darker outline, hollow point estimate, lens narrowing toward the
  estimate) and only drops the persona attribution, replacing the em dash
  with a colon.

## 3a. Decisions and Rejected Alternatives

Considered giving `rho12(fit)`'s status column an explicit word such as
"Available" or "Tested", but the Post-fit accessors table never uses status
words for available items; every other available accessor is shown by
repeating its own call (`coef(fit)` / `coef(fit)`, `nobs(fit)` / `nobs(fit)`,
etc.). Matching that existing convention was chosen over inventing a new one.
The `weights(fit)` row's "planned (parity gap)" (line 128) was left
untouched: it is a separate SHOULD-FIX row in the audit, not one of the three
BLOCKING rows in scope here, and the audit itself notes `weights(fit)` exists
but returns a placeholder (ones), so flipping it needs its own annotation,
not a one-word swap.

## 4. Files Touched

- `docs/src/rosetta.md`
- `docs/src/diagnostics-and-validation/figure-gallery.md`
- `docs/dev-log/after-task/2026-09-21-rosetta-blocking-fix.md` (new)
- `docs/dev-log/check-log.d/2026-09-21-rosetta-blocking-fix.md` (new)

## 5. Checks Run

All from the repo root of this worktree (`/private/tmp/drmodels-rosetta-fix-20260921`):

- `python3 tools/tests/test_reader_surface_audit.py`
  - Before edit: `Ran 10 tests in 1.452s` / `OK` / exit 0
  - After edit: `Ran 10 tests in 1.441s` / `OK` / exit 0
- `python3 tools/reader_surface_audit.py --public-only`
  - Before edit: `READER SURFACE AUDIT PASSED files=43` / exit 0
  - After edit: `READER SURFACE AUDIT PASSED files=43` / exit 0
- `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path = pwd())); Pkg.instantiate()'`:
  completed, `DRModels` precompiled (needed because this is a fresh
  worktree; the docs environment is not pre-instantiated).
- `env -u GITHUB_ACTIONS JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=docs docs/make.jl`:
  completed cleanly. VitePress client+server build succeeded
  (`build complete in 6.93s`), Documenter's `Doctest` / `ExpandTemplates` /
  `CrossReferences` / `CheckDocument` / `Populate` / `RenderDocument` /
  `DocumenterVitepress` stages all ran with no errors (only benign info/
  deployment-skip lines, since `GITHUB_ACTIONS`/`CI` is unset for a local
  build); the rendered `docs/build/.documenter/rosetta.md` and
  `figure-gallery.md` were inspected directly and show the edited text
  (`rho12(fit)` | `rho12(fit)`; `summary(fit)` | `summary(fit)` (prints the
  Wald coefficient table); the Confidence Eye paragraph with no "Florence").
  `Pkg.test()` was **not** run, per instructions.
- `git diff --check`: clean, no output.

## 6. Tests of the Tests

The reader-surface prose audit and its unit suite were run identically before
and after the edit (via `git stash` / `git stash pop` on the two changed docs
files) to confirm the audit's pass/fail state was not incidentally flipped by
this change; both passed in both states. That is expected: the audit checks
structural/vocabulary rules across the whole public route set, not the
specific three facts fixed here, so its own tails do not prove the fix; the
rendered-page grep above is what confirms the fix landed in the built site.
The three underlying facts (function existence, function's exact source
line, and the capabilities-page status word) were independently confirmed
against `src/` and the two other docs pages before editing (Step 1 below,
matching the audit's own cited line numbers exactly).

## 7a. Issue Ledger

- Fixed: `rosetta.md:127` claimed no `summary` method exists in Julia.
- Fixed: `rosetta.md:119` marked an implemented, tested accessor as a future
  parity gap.
- Fixed: `figure-gallery.md:7` and `:47-48` exposed an agent persona name and
  an undefined phrase ("house contract") to readers.
- Deferred: every other SHOULD-FIX row for DRModels.jl in the same audit
  (`rosetta.md:19`, `:128`, `:142`; `getting-started.md:24`;
  `model-guides/model-map.md:79`), out of scope per the maintainer's
  one-PR decision.

## 8. Consistency Audit

Cross-checked each fact against the repository before editing:

- `grep -n "function Base.summary\|rho12" src/summary.jl` → `Base.summary(fit::DrmFit)`
  at line 276, `function rho12(fit::DrmFit)` at line 117: matches the
  audit's cited line numbers exactly.
- `docs/src/getting-started.md` lines 110-160: `summary(fit)` is run directly
  in an `@example` block, and `rho12` is named in the "Post-fit accessors"
  bullet list.
- `docs/src/capabilities.md` around line 201: `` `rho12(fit)` accessor |
  **Tested** `` confirmed verbatim in the Bivariate and paired responses
  table.
- Checked every remote branch that carries unmerged work on these two files
  (surfaced by the repo's lane-check pre-edit hook) before editing: about 15
  branches touch `rosetta.md` and 1 touches `figure-gallery.md`. None of them
  touch lines 119/127 or the "Florence" text; the overlapping ones rename
  `DRModels.jl` to `DRM.jl`/`DRM`, or edit an unrelated `rho12` code snippet
  earlier in `rosetta.md` (the bivariate-formula example around line 80),
  so there is no fork of this exact fix.

## 9. What Did Not Go Smoothly

The fresh worktree's `docs/` Julia environment was not pre-instantiated;
`Pkg.develop` + `Pkg.instantiate()` had to be run first (as CI's Documenter
workflow does), adding a few minutes of precompilation before `make.jl`
itself could run. Otherwise the slice went as planned.

## 10. Known Residuals

The other SHOULD-FIX DRModels.jl rows from the 2026-09-21 cross-site reader
audit (listed in Section 7a's deferred line) are explicitly out of scope for
this PR, per the maintainer's stated one-PR decision.

## 11. Team Learning

When a rosetta/parity table has no explicit status-word column, the
convention for "available" is to repeat the Julia call itself in both
columns; introducing a new status word for a single row would have been
inconsistent with the rest of the table and easy for a future editor to
misread as a different status tier.

## 12. Review pass (2026-09-21)

Applied one SHOULD-FIX finding from the PR's code review against
`docs/src/diagnostics-and-validation/figure-gallery.md:7`: the parenthetical
had the Confidence Eye's direction backwards ("narrows from the compatibility
region to the point estimate"), contradicting the page's own account two
sections below ("widest at the point estimate and tapering to the interval
limits") and the `confidence_eye!` half-width formula
`sqrt((t - lo) * (hi - t))`, which is maximal at the estimate and zero at the
limits. "lens" and "compatibility region" were also used at line 7 before
either term is defined (definition arrives later on the page). **Applied.**
Replaced the parenthetical with: "(an interval drawn as a lens: a pale region
spanning the interval, widest at the point estimate and tapering to the
interval limits)." Lines 47-49, which already stated the direction correctly,
were left untouched. Re-ran the same four gates as the original slice:
`python3 tools/tests/test_reader_surface_audit.py` (`OK`, 10 tests),
`python3 tools/reader_surface_audit.py --public-only` (`READER SURFACE AUDIT
PASSED files=43`), the Documenter/VitePress docs build
(`env -u GITHUB_ACTIONS JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia
--project=docs docs/make.jl`, exit 0, rendered
`docs/build/.documenter/diagnostics-and-validation/figure-gallery.md`
inspected directly and shows the corrected clause with no "Florence" text),
and `git diff --check` (clean): all passed after the edit. Appended a dated
correction note to the PR body rather than replacing it, since the body's
Row 3 section repeated the same inverted "narrows toward the point estimate"
wording.
