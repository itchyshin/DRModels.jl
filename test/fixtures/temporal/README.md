# Temporal AR1 / OU fixtures (D-310)

- `ar1_gapped.csv` — 40 series, 5–9 integer occasions in 0:14 with real gaps,
  rows shuffled. Columns `y, x, id, occ`. Truth: β = (1.0, 0.5), φ = 0.6,
  σ_t = 0.8, σ = 0.5.
- `ou_irregular.csv` — 24 series, 4–7 irregular elapsed times in (0, 10), a
  stable series intercept, rows shuffled. Columns `y, x, id, elapsed`.
  Truth: β = (0.8, 0.35), λ = 0.45, σ_t = 0.65, σ_id = 0.45, σ = 0.4.

Both are simulated by `generate.jl` in this directory (StableRNGs, so they
regenerate bit-for-bit); no drmTMB code or output is involved. They exist so
the drmTMB parity cells (drmTMB#1302) can be added later by fitting the same
CSVs in R (the drmTMB calls are in the header of `generate.jl`). They are used
by `test/test_temporal_ar1.jl` and `test/test_temporal_ou.jl`.
