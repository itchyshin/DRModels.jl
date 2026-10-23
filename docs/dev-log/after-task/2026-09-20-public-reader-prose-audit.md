# After Task: Public reader prose audit

## 1. Goal

Audit every public Documenter route from a scientist's point of view, then repair
the strongest non-overlapping text defects without changing models or claims.

## 2. Implemented

- Replaced the public exact-Gaussian developer memo with a runnable Gaussian
  random-intercept example and a plain guide to `check_drm`.
- Reframed API stability around what analysts can rely on and what may change.
- Replaced internal terms such as *cell*, *admitted*, *wired*, and *payload*
  where they appeared in the owned capability and reference pages.
- Preserved all limits on unsupported models, experimental interfaces, and
  interval evidence.

## 3a. Decisions and Rejected Alternatives

The page remained in the public manual because Gaussian fit checking is useful;
only its developer-process content was replaced. Removing the page from
navigation was rejected because it would hide, rather than teach, a reader task.
Landing, beginner, migration, and model-map pages were not edited because active
documentation branches owned them.

## 4. Files Touched

- `docs/src/diagnostics-and-validation/exact-gaussian-diagnostics.md`
- `docs/src/api-stability.md`
- `docs/src/capabilities.md`
- `docs/src/reference/model-specification.md`
- `docs/src/reference/structured-effect-markers.md`
- `docs/dev-log/check-log.d/2026-09-20-public-reader-prose-audit.md`
- `docs/dev-log/after-task/2026-09-20-public-reader-prose-audit.md`

## 5. Checks Run

- Standalone execution of the new Gaussian mixed-model example: passed.
- `python3 tools/reader_surface_audit.py --public-only --landing-contract`:
  passed for all 43 public Markdown routes.
- `python3 -m unittest tools.tests.test_reader_surface_audit`: 10 tests passed.
- Expanded source search for internal identifiers and process terms across all
  43 public routes: no actionable match in the owned pages.
- `DRM_DOCS_DEPLOY=false JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia
  --project=docs docs/make.jl`: passed, including examples, cross-references,
  strict document checks, and VitePress rendering.
- `git diff --check`: passed.

## 6. Tests of the Tests

The existing reader-surface unit suite includes deliberately contaminated
temporary pages and checks that internal tracking language is rejected. The new
`@example` blocks were also evaluated by the strict Documenter build, so an
undefined function, bad field name, or failed fit would fail the build.

## 7a. Issue Ledger

- Fixed: the public exact-Gaussian page described internal evidence rows and
  promotion boundaries but gave readers no model to fit or result to interpret.
- Fixed: API stability and capability pages exposed checker and development
  vocabulary instead of consequences for an analysis.
- Deferred: reader-owned landing, beginner, migration, and model-map pages are
  covered by active branches and were deliberately not touched here.

## 8. Consistency Audit

The production navigation was parsed directly from `docs/make.jl`, yielding 43
public routes. All were searched for tracking identifiers and process terms.
The landing page already defines distributional regression, identifies
DRModels.jl as standalone Julia software, and links to a runnable first model.
The changed page was then inspected in the generated VitePress site, where its
reader-facing title appears under **Check your model**.

## 9. What Did Not Go Smoothly

The fresh clone initially lacked instantiated Julia environments. The root and
docs environments were instantiated before running the example and full build.
The repository's Graft graph was absent in the clone, so the audit used the
literal production navigation and exhaustive source search instead.

## 10. Known Residuals

Expanded `@docs` blocks can expose dense implementation language from public
docstrings even when the surrounding Markdown is clear. Those docstrings belong
to API and engine owners and were not changed in this text-only slice. The
Rosetta page contains the benign word *reconciled*; it is not an internal work
identifier and the page is owned by an active migration branch.

## 11. Team Learning

A source-only word filter can pass while a whole public page is still written as
an engineering memo. A reader audit must pair exhaustive route enumeration with
page-level reading and at least one executed path through the documented task.
The routed brain notes shaped the audience test: define the modelling question
before formulas, and keep internal evidence records out of public prose.

Memory receipt: the repository AGENTS contract, routed DRModels notes, Rose
after-task protocol, ask-brain fallback, and no-AI-slop editing rules were loaded;
the audience-first and lane-ownership rules directly shaped the file scope.

Golden Set: not in scope because this slice changed public prose only and did not
change a known code-failure class, estimator, or numerical claim.

## 12. Cross-Product Coverage

This slice covers ✓ public prose and one runnable Gaussian random-intercept
diagnostic example. It does NOT cover ✗ engine behaviour, statistical APIs,
figures, badges, package or folder renaming, R-side documentation, GLLVModels.jl,
or docstrings owned by the API and engine lanes.
