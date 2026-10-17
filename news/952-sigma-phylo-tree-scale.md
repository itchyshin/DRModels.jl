- **Gaussian `phylo(1 | g)` on `sigma`: the log-likelihood and estimates no longer
  depend on the tree height (#952).** The separate, coupled (`phylo_coupled = true`)
  and sigma-only blocks now fit on the tree rescaled to unit height, an exact
  reparameterisation. The phylo SDs are reported on the raw branch-length scale as
  before (`sd_raw = sd_unit / √height`).
  - On `ade4::carni70` (height 53.8), the default fit of
    `lsize ~ 1 + lrange + phylo(1 | species)`, `sigma ~ 1 + phylo(1 | species)`
    returned `logLik = -1e18` at its start values, and the coupled fit stopped 7.7
    nats below drmTMB.
  - Both now give the unit-height tree's values to 2e-10: -111.94868 separate, and
    -111.84106 coupled, which equals drmTMB. The ML fits take about 4-6 s instead of
    87-158 s.
  - REML, penalized (MAP) and `profile_ci` fits on that tree, which failed before,
    now match their unit-height values.
  - Unit-height trees are fitted exactly as before.
  - On other trees the route's fixed start values and boundary rules now act on the
    unit-height scale, so near a variance boundary a convergence flag or a NaN
    standard error can differ from earlier versions.
  - The default separate block has no mu-sigma phylo correlation, which drmTMB
    estimates for this formula; `phylo_coupled = true` is the drmTMB-matching model.
  - The fit-failed warning now prints `is_converged(fit)` instead of the raw
    optimiser flag.
