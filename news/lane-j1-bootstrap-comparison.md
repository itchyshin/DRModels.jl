- **Likelihood-ratio tests refuse more invalid pairs.** `lrtest`, `anova` and
  `lrt_boundary` refuse two `marginal = :VA` fits (their difference is an ELBO
  difference, #1036) and two fits with the same `nobs` but different observed
  rows (#1053). `lrt_boundary` refuses swapped arguments and a `q` larger than
  the parameters the full model adds, and both tests warn when a fit reports
  `converged = false` (#1002).
- **Bootstrap status is reported in more places.** The boundary flag now reads
  `(1 + x | g)` and phylogenetic Cholesky blocks (#1046). The marginal
  simulator refuses an ordinary random slope instead of bootstrapping an
  intercept-only model, and warns with the reason whenever it falls back to the
  conditional simulator (#997). `bootstrap_sigma_a` reports an undefined
  correlation as `NaN` rather than `0.0` and returns `prior_ridge` (#1004).
  `fit_mixed_family` returns `boot_status` and `n_boot_failed` (#1051). The q2
  bridge entry points reject unknown option keys (#1011), and `drm_bridge`
  refuses by name an option the family's `drm` does not take (#951).
