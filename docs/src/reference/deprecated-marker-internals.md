# Deprecated marker internals

!!! note "Status — Reference (parity)"
    Mirrors drmTMB's [Deprecated marker internals](https://itchyshin.github.io/drmTMB/reference/index.html) (2 in drmTMB). These are **deprecated drmTMB names**; DRModels.jl ships the modern replacement and documents the mapping here so users porting R code know what to write instead. The deprecated names are *not* part of DRModels.jl's public API.

## Deprecation map

| drmTMB (deprecated) | Use instead in DRModels.jl | Notes |
|---|---|---|
| `meta_known_V(V)` | `meta_V(v)` | Meta-analysis with known sampling (co)variances. The modern spelling is `meta_V`, attached to a `Gaussian()` model: `drm(bf(y ~ meta_V(v)), Gaussian(); data = …)`. |
| `gr(…)` | (no equivalent needed) | A drmTMB internal gradient helper; DRModels.jl computes gradients via ForwardDiff / the exact implicit-function path, so there is no user-facing equivalent. |

## Why DRModels.jl doesn't carry the old names

drmTMB keeps `meta_known_V` / `gr` only as deprecated shims for backward
compatibility with older R scripts. DRModels.jl is a fresh Julia API, so it exposes
**only the current names** — there is no legacy surface to preserve. If you are
translating an R script that calls `meta_known_V`, replace it with `meta_V`; the
arguments (the known sampling variances) carry over directly.

For the meta-analysis workflow itself, see the meta-analysis tutorial and the
`meta_V` entry in the structured-effect markers reference.
