- **Labelled (correlated) structured intercept-and-slope with SD formulas.**
  `animal(1 + x | p | g)` and `relmat(1 + x | p | g)` now fit on the Poisson,
  NegBinomial2 (`sigma ~ 1`) and Gaussian (`sigma ~ 1`) routes with
  `sd(g, animal|relmat, coef(intercept)) ~ z0` and `sd(g, animal|relmat,
  coef(x)) ~ z1`. The label adds one free correlation between the two
  unit-scale structured fields, `rho = 0.999999 tanh(eta)`, reported as block
  `:recov` (`"g:atanh_rho"`); this is drmTMB's `eta_cor_phylo` contract
  (`Cov(b0_g, b1_h) = rho s0_g s1_h C[g, h]`, `C` raw). The unlabelled form is
  unchanged (rho fixed at 0). Prediction, the bridge and the marginal
  simulator accept the labelled formula; the simulator draws correlated
  fields. Checked against independent dense Laplace / MvNormal oracles and
  against native drmTMB 0.8.1 at the optimum (one dataset per route). Other
  families (Gamma, Binomial) and labelled intercept-only structured terms
  still refuse.
