# Wave-8 disagreement evidence: what can and cannot be rerun

These scripts and tables are the record of how the wave-8 disagreements were measured. They are not part of the test suite or CI, and most cannot be rerun from this repository alone.

- Self-contained: `repro_g12.jl` (reads the one committed CSV, `data/data_binomial_ordinary.csv`). The tables `cross_eval.tsv` and `exact_md.txt` are committed outputs.
- Inputs that are not committed: `data/data_<family>_ordinary.csv` for the other families (read by `scripts/exact_md.R`), `data/data_student_ordinary.csv` (`scripts/calib2.R`), `data/coords.csv` and `data/data_gaussian_spatial.csv` (`scripts/gsp.R`), `K_fixedrange.csv` (`scripts/gsp.jl`), and `harness/run_julia.jl` (`scripts/jl_cell.jl`).
- Environment files: `scripts/env.sh`, `run_x.sh`, `run_x3.sh`, `run_x4.sh` and `run_md.sh` point at directories under the author's home (`disagree_w8_20261109`, `gapmap_w8_20261108`, `drm_campaign_20261108`, including the R library used for drmTMB 0.8.1). Those paths do not exist in the repository.

Treat the numbers in the dev-log notes that cite these scripts as recorded results, not as something a reader can regenerate here.
