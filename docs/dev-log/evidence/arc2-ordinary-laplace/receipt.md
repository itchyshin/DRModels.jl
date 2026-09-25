# Arc 2 receipt — ordinary `(1 | g)` by the TMB Laplace approximation

**Question.** Does `drm(...; marginal = :Laplace)` fit the same model as native
drmTMB (`engine = "tmb"`) on an ordinary random intercept on the mean, for
Poisson, NB2, Binomial, Gamma and Beta?

**Answer.** Yes on all ten fixtures (five families × two seeds): same df, same
log-likelihood to ≤ 3.8e-10 absolute, every free outer parameter to ≤ 6.1e-8
relative (bar: 1e-6 and 1e-5). The default `:LA` route (GHQ-32) differs from
native by 0.057 to 1.34 log-likelihood units on the same data; that gap is the
integrator, and `:LA` is unchanged. Those numbers hold only for this design
(family σ = 0.3 to 0.5, group SD 0.6). At small family σ the default `:LA` is
not slightly off but wrong: on a Gamma fixture with σ = 0.03 and a Beta fixture
with σ = 0.01 (review check, exact marginal by a mode-centred 4001-point
trapezoid), GHQ-32 at native θ̂ was −1513 and −930 against exact 304.6 and
1054.3, and the default fit reported `converged = true` at log σ = −2.393 against
−3.528 from native and `:Laplace` (σ off about 3×). `:Laplace` matched the exact
marginal to ≤ 5.2e-4 there. The `:LA` defect is not fixed by this PR.

## Files

| File | What it is |
|---|---|
| `native_fit.R` | Simulates the fixtures, writes `fixtures/*.csv`, fits `engine = "tmb"`, writes `native.tsv` |
| `julia_fit.jl` | Reads the same CSVs, fits `marginal = :Laplace` and the default `:LA`, writes `julia.tsv` and `comparison.tsv` |
| `fixtures/<family>_ri_s<seed>.csv` | The exact data both engines fitted (n = 300, 30 groups × 10, `x`, `z`, `g`) |
| `native.tsv` / `julia.tsv` | df, logLik, and every working-scale estimate per cell and engine |
| `comparison.tsv` | Per cell: df both engines, logLik both engines, abs ΔlogLik, max relative and absolute estimate difference, the `:LA` (GHQ-32) logLik and its gap, verdict |

Estimates are on the working scale both engines optimise: `mu` coefficients on
the link scale, `sigma` intercept as log σ (NB2 size = 1/σ², Gamma shape =
1/σ², Beta precision = 1/σ²), and the random-effect log-SD (`resd` in Julia,
`log_sd_mu` in TMB).

## Fixtures (DGP)

`y ~ x + (1 | g)` (plus `sigma ~ 1` for NB2/Gamma/Beta), 30 groups of 10,
`x ~ N(0, 1)`, group SD 0.6. Poisson η = 0.4 + 0.5x + b; NB2 η = 0.8 + 0.4x + b,
σ = 0.5; Bernoulli logit η = −0.2 + 0.8x + b; Gamma log η = 0.3 + 0.5x + b,
σ = 0.4; Beta logit η = 0.2 + 0.6x + b, σ = 0.3. Seeds 20260924 and 20260925.

## Result (`comparison.tsv`)

| cell | df | logLik native | abs ΔlogLik (:Laplace) | max rel Δest | abs ΔlogLik (:LA, GHQ-32) |
|---|---|---|---|---|---|
| poisson s20260924 | 3 / 3 | −491.7725798049 | 2.7e-10 | 1.6e-10 | 0.624 |
| nbinom2 s20260924 | 4 / 4 | −590.5243886827 | 3.8e-10 | 1.0e-09 | 0.081 |
| binomial s20260924 | 3 / 3 | −190.2820262535 | 1.2e-11 | 1.2e-09 | 0.173 |
| gamma s20260924 | 4 / 4 | −272.8781829274 | 1.2e-10 | 8.6e-10 | 1.092 |
| beta s20260924 | 4 / 4 | 148.3098985235 | 2.4e-11 | 7.6e-09 | 0.133 |
| poisson s20260925 | 3 / 3 | −497.1777517379 | 2.2e-10 | 1.9e-10 | 0.057 |
| nbinom2 s20260925 | 4 / 4 | −592.8996158527 | 1.5e-11 | 7.5e-11 | 0.075 |
| binomial s20260925 | 3 / 3 | −180.4241890653 | 4.8e-11 | 9.6e-10 | 0.079 |
| gamma s20260925 | 4 / 4 | −277.9981604649 | 2.9e-11 | 1.2e-10 | 1.342 |
| beta s20260925 | 4 / 4 | 164.9602689433 | 4.5e-11 | 6.1e-08 | 0.113 |

All ten: `SAME_MODEL_MATCH` (df equal, |ΔlogLik| ≤ 1e-6, max relative Δest ≤ 1e-5).

## Environment

- drmTMB: worktree `drmTMB-arc1-pr1304-fold` at `709efbeb07902fa748b124e1c268dbb8ac4ae683`
  (origin/main + folded #1304 evidence; package version 0.7.1), loaded with
  `pkgload::load_all()`; R 4.6.0, TMB 1.9.21. `OPENBLAS_NUM_THREADS=1`.
- DRModels.jl: branch `claude/arc2-laplace` (this PR's head), Julia 1.13.0,
  `JULIA_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=1`.
- Wall time: native script ≈ 20 s; Julia script ≈ 40 s including compilation.

## Reproduce

From the DRModels.jl repository root:

```sh
DRMTMB_PATH=/path/to/drmTMB Rscript docs/dev-log/evidence/arc2-ordinary-laplace/native_fit.R
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. \
  docs/dev-log/evidence/arc2-ordinary-laplace/julia_fit.jl
```

## What this does not show

- Point estimates and log-likelihood only. Standard errors, profile intervals
  and coverage were not compared here.
- Two seeds per family at one design (n = 300, 30 groups). Boundary fits (σ_b
  near 0) and near-Poisson NB2 (very large size) were not exercised; the route
  is unclamped on both scales so it follows TMB there, but no receipt shows it.
- Not covered by the route, and refused: `(1 + x | g)`, `(0 + x | g)`, crossed or
  multiple random effects, `sigma ~ covariates`, a random effect on `sigma`,
  `zi`/`hu`, structured markers (they have their own Laplace routes), and REML.
- The R bridge (`engine = "julia"`) still refuses this route under gate
  `nongaussian_mu_ordinary_random_effect`; this receipt is the Julia-side
  prerequisite for lifting it, not a bridge receipt.
