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
not slightly off but wrong: on the three banked Gamma and Beta fixtures with
family σ between 0.003 and 0.012 ("Small family σ" below, with scripts), the
default fit reported `converged = true` with a log-likelihood 295 to 501 units
below native's and σ 3.2 to 20 times native's, while `:Laplace` matched native
to ≤ 3.1e-8 in log-likelihood. The `:LA` defect is not fixed by this PR.

## Files

| File | What it is |
|---|---|
| `native_fit.R` | Simulates the fixtures, writes `fixtures/*.csv`, fits `engine = "tmb"`, writes `native.tsv` |
| `julia_fit.jl` | Reads the same CSVs, fits `marginal = :Laplace` and the default `:LA`, writes `julia.tsv` and `comparison.tsv` |
| `fixtures/<family>_ri_s<seed>.csv` | The exact data both engines fitted (n = 300, 30 groups × 10, `x`, `z`, `g`) |
| `native.tsv` / `julia.tsv` | df, logLik, and every working-scale estimate per cell and engine |
| `comparison.tsv` | Per cell: df both engines, logLik both engines, abs ΔlogLik, max relative and absolute estimate difference, the `:LA` (GHQ-32) logLik and its gap, verdict |
| `small_sigma_native.R` | Regenerates the two review fixtures `gamma_sigma0003.csv` and `nbinom2_sigma003.csv` in `test/fixtures/ordinary_laplace/`, fits all four small-sigma fixtures there with `engine = "tmb"`, writes `small_sigma_native.tsv` |
| `small_sigma_julia.jl` | Fits the same four fixtures with `marginal = :Laplace` and the default `:LA`, writes `small_sigma_julia.tsv` |
| `small_sigma_native.tsv` / `small_sigma_julia.tsv` | Native df, logLik, convergence code, max gradient, estimates; Julia logLik, converged, log σ and the gaps to native, per route |

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
| nbinom2 s20260924 | 4 / 4 | −590.5243886827 | 3.8e-10 | 7.8e-10 | 0.081 |
| binomial s20260924 | 3 / 3 | −190.2820262535 | 1.2e-11 | 1.2e-09 | 0.173 |
| gamma s20260924 | 4 / 4 | −272.8781829274 | 4.5e-11 | 1.2e-10 | 1.092 |
| beta s20260924 | 4 / 4 | 148.3098985235 | 2.4e-11 | 6.7e-11 | 0.133 |
| poisson s20260925 | 3 / 3 | −497.1777517379 | 2.2e-10 | 2.4e-10 | 0.057 |
| nbinom2 s20260925 | 4 / 4 | −592.8996158527 | 1.5e-11 | 2.5e-10 | 0.075 |
| binomial s20260925 | 3 / 3 | −180.4241890653 | 4.8e-11 | 9.6e-10 | 0.079 |
| gamma s20260925 | 4 / 4 | −277.9981604649 | 2.9e-11 | 1.2e-10 | 1.342 |
| beta s20260925 | 4 / 4 | 164.9602689433 | 4.5e-11 | 6.1e-08 | 0.113 |

All ten: `SAME_MODEL_MATCH` (df equal, |ΔlogLik| ≤ 1e-6, max relative Δest ≤ 1e-5).

## Small family σ (added after review)

Four fixtures in `test/fixtures/ordinary_laplace/`, the same files the test
suite reads. `gamma_sigma0012` and `beta_sigma0012` (family σ ≈ 0.012,
`y ~ x + z + f + (1 | g)`, simulated in R with seeds 31342 / 31343) were
already there; `gamma_sigma0003` (DGP σ = 0.003, 24 groups, n = 200) and
`nbinom2_sigma003` (near-Poisson, DGP σ = 0.03, group SD 0.5, 30 groups,
n = 246) come from the review cells (seed 99173, unbalanced groups of 4 to 13;
`small_sigma_native.R` regenerates them byte for byte). Native converged on all
four (`opt$convergence == 0`, max |gradient| at most 4.03e-4).

| fixture | logLik native | :Laplace abs ΔlogLik | :Laplace max rel Δest | :Laplace converged | default `:LA` logLik | `:LA` log σ vs :Laplace log σ |
|---|---|---|---|---|---|---|
| gamma_sigma0012 | 756.7479683642 | 3.2e-09 | 5.9e-09 | true | 289.8336876054 | −2.3658 vs −4.4297 |
| beta_sigma0012 | 1047.6527834522 | 3.0e-08 | 2.5e-07 | true | 752.9310228276 | −3.3026 vs −4.4542 |
| gamma_sigma0003 | 710.1475483826 | 3.1e-08 | 5.6e-12 | true | 209.0563586180 | −2.8398 vs −5.8398 |
| nbinom2_sigma003 | −559.5469185207 | 8.9e-12 | 4.4e-08 | true | −559.5158610133 | −2.1640 vs −2.1630 |

`:Laplace` matches native on all four (bars as above). The default `:LA`
(GHQ-32) fit reports `converged = true` on all four, but on the three Gamma and
Beta fixtures its log-likelihood is 295 to 501 units below native's and its σ
is 3.2 to 20 times native's (exp of the log σ differences 1.15, 2.06 and 3.00).
On the near-Poisson NB2 fixture the fitted σ ≈ 0.115 and `:LA` is close to
native (0.031 units).

Two defects this section closed, both on the `:Laplace` route only:

- **Near-Poisson NB2.** With the dispersion unclamped, the structured NB2
  kernel lost all precision once the size r = 1/σ² exceeded about e^20
  (`loggamma(y + r) − loggamma(r)` and `r log r − (y + r) log(r + μ)` cancel).
  The optimiser walked to log σ = −57 and reported logLik −0.0014 against
  −559.55 (`converged = false`). The route now uses its own NB2 kernel written
  without those cancellations (Stirling series for the log-gamma and digamma
  differences, `log1p(μ/r)` for the ratios), checked against a 1024-bit
  reference at r up to e^100 in the test suite. The structured NB2 kernel is
  unchanged.
- **Gamma σ = 0.003.** The route reported `converged = true` with the mean
  intercept 7.2e-5 (relative) from native. Two causes: the inner-mode tolerance
  (1e-10) left an error of about 1e-3 in the outer gradient, because the inner
  curvature is about 1e6 per group at this σ; and the Newton-decrement check
  (λ² ≤ 1e-8) accepts points about 1e-4 SE from the optimum. The route now
  solves the inner mode to 1e-13 and takes up to three Newton steps on the
  outer gradient before that check. Max relative Δest is now 5.6e-12.

## Environment

- drmTMB: worktree `drmTMB-arc1-pr1304-fold` at `709efbeb07902fa748b124e1c268dbb8ac4ae683`
  (origin/main + folded #1304 evidence; package version 0.7.1), loaded with
  `pkgload::load_all()`; R 4.6.0, TMB 1.9.21. `OPENBLAS_NUM_THREADS=1`.
- DRModels.jl: branch `claude/arc2-laplace` (this PR's head), Julia 1.13.0,
  `JULIA_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=1`.
- Wall time: native script ≈ 20 s; Julia script ≈ 40 s including compilation;
  small-σ scripts ≈ 9 s (native) and ≈ 27 s (Julia).

## Reproduce

From the DRModels.jl repository root:

```sh
DRMTMB_PATH=/path/to/drmTMB Rscript docs/dev-log/evidence/arc2-ordinary-laplace/native_fit.R
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. \
  docs/dev-log/evidence/arc2-ordinary-laplace/julia_fit.jl
DRMTMB_PATH=/path/to/drmTMB Rscript docs/dev-log/evidence/arc2-ordinary-laplace/small_sigma_native.R
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. \
  docs/dev-log/evidence/arc2-ordinary-laplace/small_sigma_julia.jl
```

## What this does not show

- Point estimates and log-likelihood only. Standard errors, profile intervals
  and coverage were not compared here.
- Two seeds per family at one design (n = 300, 30 groups), plus the four
  small-σ fixtures above. Near-Poisson NB2 is shown for one fixture whose fitted
  σ is 0.115; data whose NB2 optimum is at σ → 0 (exactly Poisson) were not
  compared with native. Boundary fits (σ_b near 0) were not compared.
- Not covered by the route, and refused: `(1 + x | g)`, `(0 + x | g)`, crossed or
  multiple random effects, `sigma ~ covariates`, a random effect on `sigma`,
  `zi`/`hu`, structured markers (they have their own Laplace routes), and REML.
- The R bridge (`engine = "julia"`) still refuses this route under gate
  `nongaussian_mu_ordinary_random_effect`; this receipt is the Julia-side
  prerequisite for lifting it, not a bridge receipt.
