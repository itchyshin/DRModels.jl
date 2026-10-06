# Class-3 boundary / estimand issues: #724, #694, #697 (2026-09-30)

Branch `claude/class3-boundary`. Measurements: Julia 1.10.12 / 1.13.1 on Totoro (one
thread per process), R 4.x + drmTMB 0.7.0 on the Mac. Scripts are reproducible from the
test files; the simulated data of #724 are embedded in
`test/test_variance_boundary_724_697.jl` (no geiger data is bundled: the package is GPL).

## #724 residual sigma at the phylogenetic boundary

Model `y ~ N(mu 1, s_a^2 A + s_e^2 I)`, n = 16 tips, intercept-only scales.

**Finding: a boundary MLE, not a flat ridge.** Engine-independent profile (dense Gaussian,
mu by GLS, `s_a^2` profiled; R, 16-tip coalescent tree with the issue's shape,
`y = 2.04 + a`, `s_e = 0`):

| `s_e` | `nll - nll_min` | `sd_phylo` (corr scale) |
|---|---|---|
| 1e-8 | 0 | 2.487958644 |
| 1e-5 | 5.9e-10 | 2.487958643 |
| 1.6e-4 | 1.5e-7 | 2.487958582 |
| 1e-2 | 5.9e-4 | 2.487712994 |
| 0.079 | 0.0386 | 2.475 |
| 1.26 | 1.80 | 1.536 |

`nll(s_e) - nll(0) ~ c s_e^2` with `c ~ 5.9`. The likelihood is **strictly increasing in
`s_e^2`**, so the MLE is `s_e^2 = 0`; it is not a flat plateau on which "any value is
optimal". What *is* flat is the working scale: `d nll / d log s_e = 2 c s_e^2` is `1.2e-7`
at `s_e = 1e-4`, below any `g_tol`, so a log-scale optimiser stops wherever the gradient
first drops under tolerance. Hence the reported `sigma_hat` is an optimiser-stopping
artefact: drmTMB 1.2e-5 and DRModels.jl 5.3e-5 on this simulation (4.3e-5 and 3.1e-4 on
carnivores), 3e-5 .. 6e-5 across 48 simulated fits. `mu`, `loglik` and `sd_phylo` agree
(loglik -28.62728 vs -28.627278; `sd_phylo * sqrt(59.2)` = 2.4879 vs R 2.4880).
Twin equality of `sigma` at this boundary is therefore not a meaningful target.

**Trigger (measured).** Ratio `r = sigma_hat_e / rms(BLUP)` (`rms(BLUP) <=` marginal
structured SD, so the diagnostic errs towards not flagging). 48 simulated phylogenetic
fits (n = 16, 64; true residual share 0 .. 0.9) plus the suite probe:
boundary fits `r <= 4.4e-5`; interior fits `r >= 2.2e-2`: a 2.7-order gap whose geometric
middle is 1e-3, the cut used (also the repo's existing sigma_b -> 0 cut in
`gaussian_ranef.jl`). Suite probe (269 Gaussian structured fits in the full suite, Julia 1.10.12): residual
ratio minimum 0.20, none flagged. Mirror trigger for the structured SD,
`rms(BLUP_k) / sqrt(s_e^2 + others)`: boundary fits `<= 1.1e-4` in simulation and
`<= 3.5e-10` in the suite (80 flagged, all plain/structured sigma_b -> 0 fits); the lowest
non-flagged value is 1.1e-3, followed by a continuum (0.002 .. 1) of genuinely small but
nonzero structured SDs (a 300-observation simulation test). So on the structured side the
cut is a *definition* (structured variance share < 1e-6), with a 5-order empty gap below it
but no gap above it; the residual side has a clean 2.7-order gap.

**Implemented.** `src/boundary_diagnostics.jl`: `_variance_boundary(fit)`; a fit-time
`@warn` from `drm(::DrmFormula, ::Gaussian)` (silent on healthy fits and in bootstrap
refits via a task-local switch); `check_drm(fit).variance_boundary`.

## #694 repeatability estimand with covariate-dependent sd(id) and sigma

`y_ij ~ N(mu_ij, s_e(z_i)^2)`, `b_i ~ N(0, s_b(z_i)^2)`, `log s_b = alpha'z`,
`log s_e = gamma'z`. The correlation of two observations of the same individual is

    R(z) = s_b(z)^2 / (s_b(z)^2 + s_e(z)^2) = logistic(2 (alpha'z - gamma'z)).

It is a function of `z`; any scalar is a choice of covariate distribution
(`E[s_b^2] / (E[s_b^2] + E[s_e^2]) != E[R]`). **Decision:** the estimand is
*conditional* at stated covariate values. The code did *not* silently pick one: it already
refused (`_variance_component_indices`), so the change is the new two-argument methods
`repeatability(fit, newdata)`, `icc(fit, newdata)`, `heritability(fit, newdata)` (the
last for `sd(g, phylogenetic)`), the precise docstring, and a refusal message that states
the formula. `logit R` is linear in the coefficients, so the Wald SE `2 sqrt(c'Vc)` (joint
vcov of the `sd` and `sigma` blocks) is exact-linear; on the tutorial data
`R(F) = 0.810 [0.721, 0.875]` (truth 0.775), `R(M) = 0.242 [0.133, 0.399]` (truth 0.308).

## #697 sigma_a vs sigma_e with one observation per species

With one observation per tip, `V = s_a^2 A + s_e^2 I` and the Fisher information for
`(s_a^2, s_e^2)` is `I = 1/2 [[tr((V^-1 A)^2), tr(V^-1 A V^-1)], [., tr(V^-2)]]`: singular
iff `A` is proportional to `I` (star tree; then only `s_a^2 h + s_e^2` is identified).
Asymptotic (Fisher) precision, median of 20 random coalescent trees, residual share 0.3:

| n | obs/tip | corr(s_a^2, s_e^2 est.) | SE(share) |
|---|---|---|---|
| 16 | 1 | -0.35 | 0.24 |
| 64 | 1 | -0.26 | 0.15 |
| 256 | 1 | -0.21 | 0.09 |
| 16 | 2 | -0.21 | 0.17 |
| 64 | 2 | -0.20 | 0.11 |

Tip-lengthened (near-star) trees, n = 16 / 64, share 0.3: mean off-diagonal tip
correlation 0.48 / 0.72 (coalescent) -> corr -0.32 / -0.31, SE(share) 0.25 / 0.15;
0.095 / 0.10 -> corr -0.96 / -0.97, SE 1.2 / 0.65; 0.029 / 0.037 -> corr -0.99, SE 3.1 / 1.2.
Finite-sample behaviour (200 simulated data sets per cell, random coalescent trees, ML):

| n | true residual share | `s_a` estimated at 0 | `s_e` estimated at 0 | interior |
|---|---|---|---|---|
| 16 | 0.1 | 0.14 | 0.07 | 0.80 |
| 16 | 0.5 | 0.51 | 0.01 | 0.49 |
| 16 | 0.9 | 0.73 | 0.01 | 0.26 |
| 64 | 0.1 | 0.01 | 0.00 | 0.98 |
| 64 | 0.5 | 0.17 | 0.00 | 0.83 |
| 64 | 0.9 | 0.58 | 0.00 | 0.41 |

**Decision: documented behaviour plus a *conditional* warning, not a blanket one.** A
warning on every one-observation-per-species fit would fire on every ordinary comparative
model (PGLS/lambda territory), against the usability rule. The boundary warning above
carries the one-observation-per-group sentence exactly when the split has collapsed
(which is what weak identification looks like in practice); an exact/near-exact star
tree is reported by the existing "Hessian numerically singular" guard (verified in the
test); `check_drm(fit).variance_boundary.one_obs_per_group` exposes the design fact.
Wald SEs of the split are unreliable (the reported vcov even had negative diagonal
entries on several near-star fits), so the tutorial points to profile / bootstrap.

## drmTMB mirror issues (for the orchestrator)

- #694 mirror: drmTMB `itchyshin/drmTMB#1244` (open, "[math] LSS personality vignette:
  define sex-specific repeatability R_i; mirror DRM.jl #694"). The `R_i` paragraph and the
  refusal cross-link belong in drmTMB's `vignettes/location-scale-scale.Rmd`.
- #697: no mirror issue found in drmTMB (searched titles/bodies for the twin wording).
  The same identification paragraph belongs in drmTMB's location-scale-scale vignette
  (phylogenetic section).
- #724: no open mirror (closest, closed: `drmTMB#1272` phylo() SD scale twin). The
  rosetta note "compare mu + logLik + structured SD, not sigma, when the residual share
  saturates" belongs in the drmTMB twin notes.
