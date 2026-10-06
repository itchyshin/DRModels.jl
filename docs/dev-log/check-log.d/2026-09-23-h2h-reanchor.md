# Check-log: Aug-24 H2H re-anchor

| Date | Slice | Command | Outcome |
|---|---|---|---|
| 2026-09-23 | H2H TMB arm | Totoro `Rscript …/run_tmb_export.R` drmTMB 0.7.1 | locscale 0.024 s; relmat 8.847 s (load~114); ll banked |
| 2026-09-23 | H2H Julia arm | Totoro `julia …/julia_from_fixtures.jl` DRModels `1b8e81c` J=4 OB=1 | locscale 0.000553 s; relmat 0.000906 s; Δll≈0 |
| 2026-09-23 | Bank | evidence dir + after-task | this PR |
