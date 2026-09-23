# Crossed-Poisson tip wall (Totoro, 2026-09-23)

Board cell `drm-crossed-poisson` (`bench/fit_crossed_poisson.jl`) re-timed on
`origin/main` tip after #781/#803. Julia arm only (no R/drmTMB pair this
receipt). Deterministic fixtures via `bench/gen_crossed_poisson.jl` (same DGP
as #70; seeds `20260710+i`).

Perspectives: Shannon (Cursor). No nested subagents.

## Protocol

| Field | Value |
|---|---|
| Tip SHA | `12ee8a8c2` (`origin/main` at run) |
| Julia | 1.12.6 · `JULIA_NUM_THREADS=1` · `OPENBLAS_NUM_THREADS=1` |
| Host | `totoro` (EPYC; leave headroom: Latte reverify + pigauto also live) |
| Command | `julia --project=. bench/gen_crossed_poisson.jl` then `julia --project=. bench/fit_crossed_poisson.jl` |
| Fit wall (script) | 25.19 s (includes compile + warm + all reps) |
| Gen wall | 1.85 s |

Headline board row uses the large-n crossed cells: `crossed_large` n=20k
median **0.1884 s**; `fixedq_n20000` n=20k median **0.1742 s**.

Machine-readable twins:

- `julia_crossed_poisson.json` (script output)
- `board_drm_crossed_poisson_20260923_12ee8a8c2.tsv`
- `fit_crossed_poisson_20260923T125726Z.log`

## Cells (Julia median)

| fixture | n | kind | median_s | logLik | conv |
|---|---:|:---|---:|---:|:---|
| single_control | 1500 | single | 0.0145 | -2266.335 | TRUE |
| crossed_small | 1000 | crossed | 0.0034 | -1382.969 | TRUE |
| crossed_medium | 5000 | crossed | 0.0308 | -7349.512 | TRUE |
| crossed_large | 20000 | crossed | **0.1884** | -29249.372 | TRUE |
| fixedq_n1000 | 1000 | crossed | 0.0126 | -1623.844 | TRUE |
| fixedq_n20000 | 20000 | crossed | **0.1742** | -28860.054 | TRUE |

Rose fence: tip absolute Julia wall only. Not a Julia-vs-Julia speedup vs the
#70 report (different host). Not vs drmTMB.
