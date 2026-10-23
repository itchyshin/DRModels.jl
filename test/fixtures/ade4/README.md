# ade4 carni70 fixture (issue #952)

- `carni70.csv` - `ade4::carni70` (Diniz-Filho & Tôrres 2002, "Phylogenetic
  comparative methods and the geographic range size-body size relationship in new
  world terrestrial carnivora", Evolutionary Ecology 16:351-367, the source ade4's
  documentation gives): 70 Carnivora. Columns: `species` (tip label, `_` replaced
  by `.`), `lsize = log(size)` (body size, kg), `lrange = log(range + 1)` centred on
  its mean.
- `carni70.nwk` - `carni70$tre`, the ultrametric Newick tree as shipped (height 53.8).

Prepared with the code in DRModels.jl issue #952 (`prep.R`), ade4 1.7-24 and
ape 5.8-1, by `docs/dev-log/evidence/952-sigma-phylo-tree-scale/core/drmtmb_native.R`.
The values agree with the CSV block posted in that issue to 1e-15: 69 of 70 rows are
textually identical, and one `lrange` value differs in its last printed digit
(Linux R 4.5.3 here, macOS R 4.6.0 there). The Newick tree is byte-identical. Only
the dataset is vendored; no ade4 code is copied. ade4's DESCRIPTION license is
`GPL (>= 2)`, and the dataset ships inside the package under it; it is also
published in the paper above. (Issue #952 cites Diniz-Filho, de Sant'Ana & Bini
1998, Evolution 52:1247, which is not the source ade4 names.) Same arrangement as `test/fixtures/lme4/README.md`.
Used by `test/test_952_sigma_phylo_tree_scale.jl`.

drmTMB reference values used there (drmTMB `ca3ddefa5ea11bbfd3b765615dfbb89305d2485a`,
TMB 1.9.23, R 4.5.3; generated outputs only, no drmTMB code copied), identical on
the raw tree and on the tree with every branch divided by 53.8:

| Model | logLik | df |
|---|---|---|
| `lsize ~ lrange + phylo(1 \| species)`, `sigma ~ 1 + phylo(1 \| species)` (native: estimates the mu-sigma phylo correlation) | -111.84106326 | 6 |
| the same objective (`f$obj$fn`) with `eta_cor_phylo` held at 0, minimised over the other 5 parameters with `nlminb` from two starts (DRModels.jl's default separate block; `nlminb` code 1 at `rel.tol = 1e-14`, max\|grad\| 1.2e-5; script `docs/dev-log/evidence/952-sigma-phylo-tree-scale/core/drmtmb_cor0.R`) | -111.94867803 | 5 |

Native fit parameters (`f$opt$par`): `beta_mu` = 1.39559462183, 0.42067158068;
`beta_sigma` = -0.01110875406; `log_sd_phylo` = -0.68444385693 (mu),
-0.14630323781 (sigma), on the unit-tip-variance scale; `eta_cor_phylo` =
0.24334775573, i.e. cor = 0.999999 tanh(eta) = 0.2386551.
