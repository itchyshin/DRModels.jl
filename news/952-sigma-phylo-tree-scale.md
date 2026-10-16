- **Gaussian `phylo(1 | g)` on `sigma` no longer depends on the tree height (#952).**
  The separate, coupled (`phylo_coupled = true`) and sigma-only blocks now fit on
  the tree rescaled to unit height and report the phylo SDs on the raw
  branch-length scale as before (`sd_raw = sd_unit / √height`). On
  `ade4::carni70` (height 53.8) the default fit of
  `lsize ~ 1 + lrange + phylo(1 | species)`, `sigma ~ 1 + phylo(1 | species)`
  returned `logLik = -1e18` at its start values, and the coupled fit stopped
  7.7 nats below drmTMB; both now give the unit-height tree's values to 1e-9
  (-111.94868 separate, -111.84106 coupled = drmTMB), in about 5 s instead of
  90-160 s. Unit-height trees are fitted exactly as before. The default
  separate block has no mu-sigma phylo correlation, which drmTMB estimates for
  this formula; `phylo_coupled = true` is the drmTMB-matching model. The
  fit-failed warning now prints `is_converged(fit)` rather than the raw
  optimiser flag.
