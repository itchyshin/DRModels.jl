# Gaussian REML routes: fixtures

Committed: the simulated datasets used by the test (`ord_s*.csv`, `crossed_s*.csv`, `animal_s*.csv`, `relmat_s*.csv`, `phylo_s*.csv`), the relationship matrix (`A.csv`, `A_ids.csv`), the tree (`tree.nwk`), and R drmTMB 0.8.1 `REML = TRUE` output (`r_reml_reference.csv`). `compare.jl` and `test/test_gaussian_reml_routes.jl` run from these alone.

Not committed: `gen_ref.R` also reads base covariate files (`data_gaussian_animal.csv`, `data_gaussian_relmat.csv`, `data_gaussian_phylo.csv`) and loads drmTMB from an R library under the author's home directory, so it cannot be rerun from the repository. The committed CSVs and reference are its real outputs; a local R `REML = TRUE` refit on the committed data reproduced every reference logLik to 7e-13 (review, 2026-11-09).
