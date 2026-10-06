# Moving older R code to DRModels.jl

If you are translating an older `drmTMB` analysis, use the modern spelling in
the table below. DRModels.jl does not carry the retired R names.

## Deprecation map

| drmTMB (deprecated) | Use instead in DRModels.jl | Notes |
|---|---|---|
| `meta_known_V(V)` | `meta_V(v)` | Meta-analysis with known sampling (co)variances. The modern spelling is `meta_V`, attached to a `Gaussian()` model: `drm(bf(y ~ meta_V(v)), Gaussian(); data = …)`. |
| `gr(…)` | (no equivalent needed) | This R helper has no user-facing Julia equivalent. |

## Why the older names are not available

drmTMB keeps `meta_known_V` / `gr` only as deprecated shims for backward
compatibility with older R scripts. DRModels.jl is a fresh Julia API, so it exposes
**only the current names** — there is no legacy surface to preserve. If you are
translating an R script that calls `meta_known_V`, replace it with `meta_V`; the
arguments (the known sampling variances) carry over directly.

For the meta-analysis workflow itself, see the meta-analysis tutorial and the
`meta_V` entry in the structured-effect markers reference.
