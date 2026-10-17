# Speed per family — DRM.jl vs drmTMB, consolidated

*Assembled 2026-08-27 (completion-roadmap Wave D, issue #9's "speed edge documented per
family"). Every number below is a previously measured, committed result or a fresh
measurement dated today — sources cited per row. No number here is extrapolated.*

**Reading rules.** All engine-vs-engine rows compare **warm fit medians** on the same
model; the two caveat rows at the bottom are part of the claim, not footnotes. Speedup
= R median / Julia median; >1 means DRM.jl is faster. New rows follow the labelling
rules in [`docs/design/timing-protocol.md`](../docs/design/timing-protocol.md)
(backend, workers/threads, BLAS and OpenMP threads, platform).

## The table

| family / route | cells | R median | Julia median | speedup | source |
|---|---|---|---|---|---|
| **Gaussian q4 location-scale** (phylo, p=100, real data) | 1 | 2.48 s | 1.14 s | **2.18×** | `comparison-grid.md` (2026-05-30, verified; drmTMB reported false-convergence on the same cell) |
| **Poisson phylo** (mean intercept, p=100…2000) | 4 | 0.079–1.52 s | 0.044–0.80 s | **1.65×** med (1.39–1.89) | `phylo-poisson-benchmark.md` |
| **NB2 phylo** | 4 | — | — | **58.1×** med (28.2–71.6) | `phylo-nb2-benchmark.md` |
| **Binomial phylo** (n=72 toy) | 1 | 40 ms | 1.6 ms | **~24×** | fresh 2026-08-27 (see note ①) |
| **Gamma phylo** (n=72 toy) | 1 | 57 ms | 3.5 ms | **~16×** | fresh 2026-08-27 ① |
| **Beta phylo** (n=72 toy) | 1 | 76 ms | 5.8 ms | **~13×** | fresh 2026-08-27 ① |
| **Crossed Poisson** (two crossed intercepts) | 6 | — | — | **18.2×** med (10.8–35.8) | `crossed-poisson-benchmark.md` (#70) |
| **Crossed non-Gaussian families** | 6 | — | — | **41.5×** med (25.5–175.2) | `crossed-family-benchmark.md` (#80) |
| Gaussian location-only (conjugate EM vs own LBFGS) | 1 | n/a | 0.18 s vs 0.57 s | 3.1× *(internal)* | `comparison-grid.md` — not an R comparison |

① The old `phylo-binomial-benchmark.md` / `phylo-gamma-beta-benchmark.md` files predate
drmTMB's native support for these routes (gained 2026-08-17 for binomial phylo) and
carry Julia-only timings. The fresh numbers use the `parity_phylo_nongaussian.R` fixture
recipe (12 tips × 6, unit-height tree), warm medians of 5 per engine, same DGP on both
sides (not byte-identical draws — timing only; *parity* numbers always come from the
byte-identical harnesses). Toy-scale n: these three ratios measure per-fit overhead, not
asymptotic scaling.

## The two honest caveats

- **First-call latency.** Through `engine = "julia"` from R, the first fit pays one-time
  JuliaCall startup + JIT compilation — measured 27.7 s (p=1000) and 21.9 s (p=3000) *total
  wall*, dominated by startup, not fit cost (`2026-08-24-phylo-large-p-probe.md`: less
  total time at the larger p). A one-shot R user experiences that latency; the table's
  warm medians describe every fit after the first, and native-Julia users skip it entirely.
- **Where the comparison is even.** On the Poisson-phylo route the edge is modest
  (1.4–1.9×) and native TMB scaled O(p^1.27) over the measured range (DRM.jl#486) — the
  dense-cubic premise is measurably false there, and speed claims against it should stay
  route-specific. The large speedups (NB2, crossed routes) are where TMB's own overhead
  per family is highest, not a universal constant.

## Relation to the O(p) claim

DRM.jl's own O(p) scaling to p=10,000 is documented in `qgate-multishape-scaling.md` /
`q4-sparse-status.md` and is a **single-engine** property; no native comparator has been
attempted near that scale (recorded on the `phylo_count_large_p` ledger row). This table
deliberately does not repeat it as a versus-R claim.
