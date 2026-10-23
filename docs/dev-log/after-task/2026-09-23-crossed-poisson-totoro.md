# After-task: drm-crossed-poisson Totoro tip wall (2026-09-23)

## Scope

Fill the last DRModels `needs_run` cell on the three-package speed board:
`drm-crossed-poisson` via the existing #70 script (no new DGP).

## Outcome

- Tip `12ee8a8c2` on Totoro, threads=1 / BLAS=1.
- Gen 1.85 s; fit script wall 25.19 s.
- `crossed_large` (n=20k) median **0.1884 s**; `fixedq_n20000` median **0.1742 s**.
- Receipt banked under `docs/dev-log/evidence/2026-09-23-crossed-poisson-totoro/`.

## Soft follow-ups (deferred)

Tip p5000 re-time and Aug-24 H2H re-anchor left for a quieter Totoro window
(load ~144 with Latte reverify + pigauto). Not started this slice.

## Rose

Tip absolute only. Host = Totoro EPYC ≠ Mac Studio #70/#803 walls. No speedup
column claimed.
