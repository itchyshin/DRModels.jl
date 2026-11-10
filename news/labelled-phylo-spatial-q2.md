- **Labelled (correlated) phylogenetic and spatial intercept-and-slope with SD
  formulas.** `phylo(1 + x | p | g)` and `spatial(1 + x | p | g)` now fit on the
  Poisson, NegBinomial2 (`sigma ~ 1`) and Gaussian (`sigma ~ 1`) routes with
  `sd(g, phylogenetic|spatial, coef(intercept)) ~ z0` and `sd(g, ..., coef(x))
  ~ z1`, adding one free correlation (`rho = 0.999999 tanh(eta)`, block
  `:recov`, `"g:atanh_rho"`) to the unlabelled q2 routes. Kernels are the
  native ones (`ape::vcv(tree, corr = TRUE)`; `exp(-d / median d) + 1e-6 I`).
  Prediction, the bridge and the marginal simulator accept the labelled form.
  Checked in this repository against dense Laplace / MvNormal oracles. Where
  native drmTMB 0.8.1 fits the same shape, the comparison at the optimum was
  run on the R side and its numbers are not stored here; the evidence is in
  itchyshin/drmTMB PR 1684, folder `wave3-labelled-phylo-spatial`. Native
  drmTMB 0.8.1 refuses SD formulas on labelled Poisson phylo and any labelled
  NB2 spatial term; those two cells are validated against the oracle only.
