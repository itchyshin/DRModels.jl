# Receipt: `marginal = :Laplace` for the Gaussian random intercept on sigma

Date: 2026-09-25. Branch `claude/arc2-sigma-re-laplace` on top of DRModels.jl
`da8b3f871`. Native side: drmTMB `4902180c0` (0.7.1, lane
`drmTMB-arc1-pr1304-fold`, `pkgload::load_all`, `engine = "tmb"`), TMB 1.9.21,
R 4.6.0. Julia 1.13.0, `JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`.

## Model

`bf(y ~ x, sigma ~ 1 + (1 | g))` (cell c6: `sigma ~ 1 + x + (1 | g)`),
Gaussian: `log sigma_i = X_sigma beta_sigma + b_g`, `b_g ~ N(0, sd_b^2)`.
drmTMB integrates `b` by TMB's Laplace approximation (`u_sigma ~ N(0, 1)`,
`b = sd_b u`). DRModels' default (`marginal = :LA`) integrates it by
non-adaptive 32-node Gauss-Hermite quadrature. The new `marginal = :Laplace`
computes, per group, `h(b_hat) + 0.5 log(2 pi) - 0.5 log(-h''(b_hat))` with
`h(b) = sum_i log N(y_i; mu_i, exp(eta0_i + b)) + log N(b; 0, sd_b^2)`.
Laplace is invariant to the affine change `b = sd_b u`, and the groups are
independent, so this is TMB's value.

## Scripts and outputs (this folder)

| file | role |
|---|---|
| `native_fit.R` | simulates the seven fixtures, writes `fixtures/*.csv`, fits drmTMB, writes `native.tsv` |
| `julia_fit.jl` | fits every fixture with `:Laplace` and the default, writes `julia.tsv` and `comparison.tsv` |
| `exact_integral.jl` | exact marginal (adaptive quadrature per group) vs Laplace vs GHQ-32 at native estimates, writes `exact_integral.tsv` |
| `d273_default_unchanged.jl` | default-route fit of every fixture; output `d273_default_unchanged.txt` |

Cells p1 to p5 are the conductor's probe designs and seeds
(`sigre-ghq-probe.R`); the native logLik column reproduces the probe's native
column to the printed digits. c6 adds a covariate on sigma; c7 has 20 groups of
3 to 120 rows.

## Native match (`comparison.tsv`)

| cell | n | G | df (tmb/jl) | logLik native | abs dlogLik | max rel d est | max rel d SE | converged | default GHQ-32 minus native |
|---|---|---|---|---|---|---|---|---|---|
| p1 15 x 10, sd_b 0.40 | 150 | 15 | 4/4 | -189.7990458468 | 1.9e-11 | 1.6e-10 | 1.8e-06 | true | +0.0321 |
| p2 10 x 50, sd_b 0.40 | 500 | 10 | 4/4 | -614.0428104368 | 2.6e-10 | 2.9e-10 | 1.5e-06 | true | +0.2986 |
| p3 8 x 150, sd_b 0.40 | 1200 | 8 | 4/4 | -1302.1142745893 | 8.4e-11 | 1.5e-10 | 1.5e-06 | true | +0.4806 |
| p4 8 x 150, sd_b 0.15 | 1200 | 8 | 4/4 | -1396.2470571756 | 3.4e-11 | 1.8e-10 | 6.6e-07 | true | +0.0019 |
| p5 6 x 400, sd_b 0.30 | 2400 | 6 | 4/4 | -2787.0591192645 | 1.7e-10 | 1.4e-10 | 1.4e-06 | true | +0.3397 |
| c6 sigma ~ 1 + x, 12 x 60 | 720 | 12 | 5/5 | -864.6322991303 | 4.7e-11 | 1.7e-10 | 2.0e-06 | true | -0.2940 |
| c7 unequal, 20 groups | 1172 | 20 | 4/4 | -1406.5498135490 | 3.5e-11 | 2.5e-10 | 1.5e-06 | true | -3.2573 |

All seven: df equal, |dlogLik| <= 1e-6, every working-scale estimate within
1e-5 relative (observed <= 2.9e-10), `converged = true`. SEs: TMB's
`sdreport` `cov.fixed` against the ForwardDiff Hessian of the Laplace
objective, within 2e-6 relative. Native `opt$convergence == 0` and
max |outer gradient| <= 6.3e-9 in every cell.

Note on p5: the probe reported the Julia default (through the drmTMB bridge)
as +2.674 from native. At this DRModels head the default route gives +0.3397,
both directly and through `drm_bridge` (checked in Julia). The probe's Julia
checkout was not identified; not investigated further.

## Which integrator is closer to the exact marginal (`exact_integral.tsv`)

Evaluated at native drmTMB's estimates; exact = per-group adaptive quadrature
over +-12 posterior SDs around the mode (rtol 1e-12).

| cell | Laplace minus exact | GHQ-32 minus exact |
|---|---|---|
| p1 (10 rows/group) | -0.0310 | 0.0000 |
| p2 (50) | -0.0249 | -0.0745 |
| p3 (150) | -0.0074 | -0.4872 |
| p4 (150, sd_b 0.15) | -0.0019 | -0.0000 |
| p5 (400) | -0.0024 | -4.7813 |
| c6 (60, sigma ~ x) | -0.0262 | -0.1060 |
| c7 (3 to 120) | -0.0585 | -1.6888 |

So the gap to native is not a Julia defect in the Laplace sense: GHQ-32 is
exact for small groups, and Laplace (TMB) carries its usual O(1/m) error there.
For large groups the fixed nodes are too coarse for the narrow per-group
integrand and GHQ-32 is the one far from exact. `:Laplace` is the same-target
answer for drmTMB parity; it is not claimed to be the more accurate integrator
for small groups.

## Default path unchanged (D-273)

`d273_default_unchanged.jl` run under `git archive da8b3f871` (with this
worktree's `Manifest.toml`) and under this branch printed byte-identical
output (`cmp` clean): logLik, `repr(theta)`, `converged`, `hash(vcov)` and
`marginal = :LA` for all seven fixtures. The committed
`d273_default_unchanged.txt` is that output.

## Tests

`test/test_sigma_re_laplace.jl` (registered in `test/runtests.jl`): native
match on three committed fixtures (p1, c6, c7; `test/fixtures/sigma_re_laplace/`,
the file's only cross-engine constants); objective equal to an independent
Laplace reference written with Distributions + ForwardDiff (1e-9, at the
optimum and at a perturbed point); ForwardDiff gradient and Hessian through
the inner mode against central differences; the default equals an explicit
`marginal = :LA` / `:la` fit bit for bit and equals an independent GHQ-32
transcription at its optimum; Laplace against a dense 1-D integral for one
group (error < 2e-3 at 400 rows, more than 5 times larger at 25 rows); the
bracketed mode solver on extreme inputs; refusals; and the `drm_bridge`
`marginal` option and `"marginal"` output.
