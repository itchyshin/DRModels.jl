# q4 phylo p=5000 tip wall (Totoro, 2026-09-23)

Board cell `drm-gauss-q4-phylo-p5000` (`bench/profile_q4_sections.jl --gate tsv --p 5000`)
re-timed on `origin/main` tip after the Mac mid-flight kill left tip absolute owed.

Perspectives: Shannon (Cursor). No nested subagents.

## Protocol

| Field | Value |
|---|---|
| Tip SHA | `12ee8a8c2` (`origin/main` at run) |
| Julia | 1.12.6 · `JULIA_NUM_THREADS=4` · `OPENBLAS_NUM_THREADS=1` |
| Host | `totoro` (EPYC 9655; load ~144/384 at start; Latte + pigauto live) |
| Command | `julia --project=. bench/profile_q4_sections.jl --gate tsv --p 5000` |
| Gate | G5e.3 PASS |

Headline: p=5000 chosen-rep fit wall **82.4455 s** (reps_used=3; chosen_rep=1).
`fd_vcov` projected wall 204.0054 s (not the board headline).

Machine-readable twins:

- `q4_sections_12ee8a8c2.tsv` (script output; also under `bench/results/`)
- `board_drm_gauss_q4_phylo_p5000_20260923_12ee8a8c2.csv`
- `profile_q4_sections_p5000_20260923T130505Z.log`

Rose fence: tip absolute Julia wall on Totoro only. Not a Julia-vs-Julia speedup
vs Mac banks (`a734d2b90` / `9d709f008` / `cf058168b`). Not vs drmTMB.
