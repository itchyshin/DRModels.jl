# After Task: DRM.jl → DRModels.jl package rename

## 1. Goal

Prepare the Julia package rename from `DRM.jl` to `DRModels.jl` on a reviewable,
unmerged branch. Preserve the public modelling API (`drm`, `bf`, fit and post-fit
functions), UUID, and version; do not rename the GitHub repository, merge the PR,
touch the R bridge, publish a registry release, or rewrite historical dev-log files.

## 2. Implemented

- Renamed the package, module, source entry point, and Makie extension to
  `DRModels` / `src/DRModels.jl` / `DRModelsMakieExt`.
- Rewrote current shipping imports, qualified module calls, tests, benchmarks,
  tools, Documenter configuration, README, citation metadata, workflows, and NEWS.
- Kept the modelling API unchanged. `DRModels.DRM` is a deliberately deprecated
  qualified alias; `using DRM` cannot survive a Julia package rename.
- Kept the UUID and package version unchanged.
- Kept existing `docs/dev-log/` history unchanged; this report and its matching
  check-log row are new records.

## 3a. Decisions and Rejected Alternatives

The source-level alias is retained only as `DRModels.DRM`; attempting to preserve
`using DRM` would misrepresent Julia's package resolver. The GitHub repository name,
registry identity, drmTMB bridge, stats-hours notebook imports, R releases, and book
republishing were explicitly rejected from this branch because they are coordinated
Shinichi gates. Aqua's temporary-project persistent-task probe is deferred until the
post-rename registry identity exists: it cannot instantiate an unregistered package
name using the UUID still registered as `DRM`; all other Aqua checks stay enabled.

## 4. Files Touched

The branch changes 480 files: `Project.toml`; `src/DRM.jl` renamed to
`src/DRModels.jl`; `ext/DRMMakieExt.jl` renamed to `ext/DRModelsMakieExt.jl`; 293
test files; 52 current Documenter pages; 27 benchmark files; 59 tools; the current
README, `CITATION.cff`, `NEWS.md`, documentation configuration, and CI metadata.
The exact enumerated manifest is the PR diff for #773. This report and
`docs/dev-log/check-log.d/2026-09-17-drmodels-package-rename.md` are the only new
dev-log files; no historical dev-log path changed.

## 5. Checks Run

- Totoro: `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` — PASS (`Testing DRModels tests passed`).
- Local Documenter build — PASS.
- GitHub Actions run `35260107842`: Julia 1.10 and current Julia, each four-way
  sharded — all eight shards PASS.
- GitHub Actions Documenter run `35260107737`: `docs` and `docs-preview` — PASS.
- Scoped residual audit over `src`, `test`, `docs/src`, `bench`, and `tools` — only
  the intentional alias/migration text and a historical benchmark output retain a
  standalone `DRM` spelling.

## 6. Tests of the Tests

The full matrix loads the renamed package by its new name and exercises every shard.
`test/runtests.jl` asserts `DRModels.DRM === DRModels`, so removal or miswiring of
the explicitly approved source-level migration alias fails. The two repaired r2
controls qualify the exported `Poisson` constructor through `DRModels`, preventing
the shared-test-module name ambiguity that appeared during the mechanical rename.

## 7a. Issue Ledger

PR [#773](https://github.com/itchyshin/DRM.jl/pull/773) contains this Arc 0 rename
and remains open and unmerged. A transient current-Julia q2 REML `PosDefException`
on an earlier head was independently audited: it is a separate numerical-admissibility
issue, not rename-caused, and the final matrix passed that shard unchanged. It should
be filed separately if it recurs; no engine work was folded into this rename.

## 8. Consistency Audit

The package/module filename, extension declaration, imports, qualified calls,
Documenter module list, sitename, repository/Pages staging URLs, README installation
path, citation, NEWS, tests, tools, and CI labels were checked together. `DRM_JL_PATH`
and other bridge-facing legacy environment variable names remain intentionally stable
while the local folder stays `DRM.jl`, as agreed in the Arc Card. The source tree,
tests, and current docs have no accidental standalone old-module reference.

## 9. What Did Not Go Smoothly

Two test controls initially referred to unqualified `Poisson` in the shared test
module and needed `DRModels.Poisson()`. Aqua's persistent-task subprocess then
exposed General's still-old UUID/name record; adding transitive packages one at a
time was rejected and the single inapplicable probe was deferred with an explanation.
An earlier current-Julia q2 REML run threw `PosDefException`, but a separate
read-only audit and the final green current-Julia shard showed it was not introduced
by this rename.

## 10. Known Residuals

This PR deliberately does NOT rename the GitHub repository, merge to `main`,
register/publish the package, modify drmTMB or gllvmTMB, alter stats-hours, release
any R package, or republish the book. The Aqua persistent-task test remains deferred
until the package has its approved post-rename registry identity. Historical dev-log
records preserve their original names by design.

## 11. Team Learning

The package-name migration has a hard Julia boundary: a const alias can help code
already using `DRModels`, but cannot retain `using DRM`. A current General registry
record with the same UUID under the old name also makes Aqua's fresh-project
persistent-task test invalid during the pre-registration interval. The Arc Card and
working document were consulted for these boundaries; no durable vault decision or
memory record was changed in this PR.

**Memory receipt:** loaded the approved Arc Card and its DRM working document, then
used their explicit boundaries (current shipping files change; existing dev-log
history, registry, R bridges, releases, and GitHub repository rename do not). No
additional memory rule was discovered that merits a durable update.

Golden Set: the rename does not alter an estimator, formula grammar,
model family, or bridge contract, so no project Golden-Set fixture is applicable.
The full Julia suite on Totoro and both supported CI Julia versions is the relevant
regression evidence for this package-identity transformation.

## 12. Cross-Product Coverage

The rename covers ✓ the Julia package identity, source entry point, extension,
shipping Julia callers, current Documenter configuration, package metadata, and both
supported CI Julia versions. It does NOT cover GitHub repository identity, Pages
deployment at the new live URL, the Julia General registry, the drmTMB/gllvmTMB R
bridges, stats-hours notebooks, R releases, or book republishing. Those downstream
surfaces remain explicit coordinated gates rather than partial changes hidden in a
Julia source rename.
