- **Multi-start for Gamma coordinate-SD (`sd-z`) q2 fits.** The Gamma two-field
  route (`phylo`/`spatial`/`animal`/`relmat(1 + x | g)` with an SD formula that
  has a non-intercept predictor) has a second interior mode in the SD
  coefficients, and the single default start stopped at the lower one in 2 of 22
  wave-9 seeds (spatial seeds 9 and 10, 0.46 and 1.41 below drmTMB). Such fits now
  run the six alternative starts of the shared q2 restart helper and keep one only
  on a clear objective gain. The outcome is in `fit.scales[:q2_restarts]`
  (ran, winning start, gain, convergence flipped). Cost: a fit takes about 6 times
  as long (median 1.5 s to 9.7 s on the 22 seeds). Gamma fits with constant SDs
  are unchanged.
