# DRAC first-wave DRM receipts (Fir)

**Job:** `61148082` (`drm-fw`, array 1–10%5) on Fir. Predecessor `61144759` failed on concurrent
`Pkg.instantiate`; resubmitted with shared scratch depot `/scratch/snakagaw/julia_depot_speed_phaseB`.

**Outcome:** 10/10 COMPLETED batch exit 0, but **every cell wrote `status=error`** (MethodError on
`drm(...; tree=, se=)` / related kwargs). Combined: `drm_fir_61148082_combined.tsv`.

**Board authority for Phase B cells 11–20:** Totoro tip-abs CSV
`docs/dev-log/evidence/2026-09-23-speed-kinds-toward20/board_drm_first_wave_20260923_e9d50a110.csv`
(job-local harness `run_phylo_repair.jl` / `run_first_wave.jl` @ `e9d50a110`). Fir DRAC harness
`bench/speed_board_firstwave_drm.jl` still needs an API repair before a second Fir campaign.

Julia abs only this wave (TMB H2H pair deferred). Do not form Fir×Totoro speedups from these rows.
