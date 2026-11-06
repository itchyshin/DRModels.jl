# Contributing to DRModels.jl

Notes for people working on the package itself. Users do not need them.

- [HANDOVER.md](HANDOVER.md) records engineering provenance,
  [ROADMAP.md](ROADMAP.md) the development plan, and [AGENTS.md](AGENTS.md)
  the review roles and the Definition of Done.
- `src/` is the engine and public API; `src/experimental/` holds unwired
  prototypes and recorded negative results (per-file verdicts in
  [src/experimental/README.md](src/experimental/README.md)); `bench/` holds
  the runnable benchmarks and fixtures; `test/` the test suite; `report/` the
  design, provenance and benchmark reports; `docs/` the Documenter site
  (DocumenterVitepress) and its [CONTRACT.md](docs/CONTRACT.md).
- **Versions and registration.** The `v0.7.1` git tag predates the rename
  from `DRM` to `DRModels` and names the old package, so it is not a DRModels
  release tag. Registration in the Julia General registry is deferred until
  both DRModels.jl and drmTMB are ready; until then install via
  `Pkg.add(url = ...)`.
- The drmTMB comparison tests run only when `DRM_PARITY_TESTS=1` is set
  (RCall against fixtures generated with drmTMB v0.7.0); families are
  otherwise checked by simulation-based parameter recovery.
- To develop from a local clone
  (`git clone https://github.com/itchyshin/DRModels.jl`), use
  `Pkg.develop(path = "/absolute/path/to/DRModels.jl")` instead of `Pkg.add`.
