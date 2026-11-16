- **Post-fit methods on the wave-2 SD-regression fits.** `predict`,
  `predict_parameters`, `simulate` (default `re = :marginal`),
  `bootstrap_result` and `drm_bridge` now work on the Gamma/NegBinomial2
  two-field SD fits (same-group `(1 | g) + (0 + x | g)`, crossed
  `(1 | g) + (1 | h)`, ordinary + `relmat`/`animal(1 | p)`, Gamma
  `relmat(1 + x | g)`, labelled `animal`/`relmat(1 + x | p | g)`, NB2
  `(1 + x | g)`) and on the phylo / coordinate-spatial `(1 + x | g)` q2 fits
  (Gaussian, Poisson, NegBinomial2). Population-level prediction is
  `g^{-1}(X beta)`; the label `p` in `kind(1 + x | p | g)` is a pure label and
  is never read as a data column. The marginal simulator draws fresh latent
  fields from the fitted SD formulas and kernel (correlated when the fit has
  `rho`); phylo/spatial simulation needs the fitting `tree` / `coords`.
  The two-field correlation is now named `"g:atanh_rho"` like the other
  labelled q2 routes.
- `ranef(fit)` now refuses (as it already did for `:sd`, `:resd` and `:recov`
  blocks, #759) on any SD-regression fit (`sdtarget_*` blocks) that stores no
  conditional modes, instead of returning an empty `Dict`.
- Known limitation: `residuals(fit; type = :quantile)` on SD-regression fits
  uses the fixed-effect-only reference and does not integrate the random
  effects out, so the residuals are over-dispersed when the random-effect SDs
  are non-negligible. This matches the earlier Poisson/NB2/Gaussian q2 routes;
  use `simulate(fit; data, …)` for calibration checks.
