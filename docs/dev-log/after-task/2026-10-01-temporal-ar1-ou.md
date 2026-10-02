# After-task: temporal AR1 / OU random effects, wave 1 (2026-10-01)

## Scope

Owner decision D-310: the Julia twin of drmTMB's
`temporal(1 | id, time = occ, structure = "ar1" | "ou")` on the Gaussian mean.
Wave 1: Gaussian family, `mu` only, one unlabelled intercept, `sigma ~ 1`, ML,
an optional ordinary `(1 | id)` on the same id (as drmTMB). Branch
`claude/temporal-ar1-ou` from origin/main d9c2afccf. Refs drmTMB#1302.

Semantics were read (not copied) from drmTMB branches
`claude/lane-temporal-ar1-v2` and `codex/temporal-ou-v1-20260908`
(`R/temporal.R`, `R/parse-formula.R`, `R/drmTMB.R`, the vignette). drmTMB is
GPL; no drmTMB source or output is in this PR.

## What changed

- `src/temporal.jl` (new): the `temporal` marker, the term parser, drmTMB's
  data checks (complete id/time, finite numeric time, integer AR1 occasions,
  unique `(id, time)`, the distinct-lag requirement — 2, or 3 with `(1 | id)`,
  and an odd lag for AR1 — and multiple series with `(1 | id)`), the engine,
  the scope router, and `temporal_parameters(fit)`.
- Engine: per series, a scalar Kalman filter (prediction-error
  decomposition) gives the exact `logdet V` and `rᵀV⁻¹r` in O(n); the
  ordinary intercept is a rank-one update computed by filtering the vector of
  ones with the same gains; conditional temporal effects come from the RTS
  smoother. A tridiagonal-precision evaluation (Q = R⁻¹) was implemented first
  and replaced: Q has entries 1/(1 − a²), and an optimiser line search into
  λ → 0 produced a negative `1ᵀV⁻¹1` (DomainError) on the OU recovery data.
  The filter has no such cancellation; 1 − a² is formed as `-expm1(-2λΔt)`
  (OU) and `sech²θ Σ φ^{2k}` (AR1).
- `src/gaussian_ranef.jl`: `_split_ranef` returns the temporal term in a 6th
  slot only when the caller opts in (`allow_temporal = true`); every other
  caller (all non-Gaussian families, the `sigma` formula, bivariate, mixed
  family, Laplace/AGHQ Gaussian, bootstrap) gets a refusal naming the scope.
- `src/gaussian_core.jl`: the temporal route is dispatched first in
  `_drm_gaussian_fit`; `predict` opts in (population-level `Xβ`, as for every
  structured marker in DRModels).
- `src/bridge.jl`: `temporal` joins the DSL calls; drmTMB's keyword spelling
  is rewritten to the positional `@formula` form, with drmTMB's argument checks.
- `src/introspection.jl`, `src/summary.jl`: `structured_effects` lists the
  term; block titles/null notes; `profile_targets` scale `:atanh` (AR1) /
  `:log` (OU decay).
- Exports `temporal`, `temporal_parameters` (EXPERIMENTAL tier in
  `test/test_api_stability.jl` and `docs/src/api-stability.md`); reference
  docs, capabilities row, news fragment.
- Tests `test/test_temporal_ar1.jl`, `test/test_temporal_ou.jl`; fixtures and
  generator in `test/fixtures/temporal/`; GLLVModels.jl cross-check receipt in
  `docs/dev-log/evidence/temporal-ar1-ou/`.

## Evidence

- Dense oracle (−logpdf of the dense `MvNormal`): |Δnll| < 1e-10 at 4 random
  θ plus 2 edge θ on each of 12 data sets (AR1 and OU, ± `(1 | id)`, unequal
  series, gapped/irregular times, shuffled rows); rtol 1e-8 at φ = tanh 9 and
  λ = e⁻²⁰. Conditional modes match `σ_t² R V⁻¹ r` / `σ_b² Zᵀ V⁻¹ r` to 1e-8.
- GLLVModels.jl (origin/main 5712e35d5) `fit_temporal_gllvm`, three stacked
  identical traits (its port admits ≥ 3 traits; by symmetry logLik = 3 ×):
  |ΔlogLik| = 1.1e-13 (AR1 fixture), 0 (OU fixture), 2.8e-14 (OU + `(1 | id)`).
- Recovery (S = 800 series): AR1 φ̂ 0.676 (0.6), σ̂_t 0.758 (0.8), σ̂ 0.560
  (0.5); OU λ̂ 0.466 (0.45), σ̂_t 0.633 (0.65), σ̂ 0.389 (0.4). A 40-replicate
  Monte Carlo of the AR1 DGP at S = 300: mean φ̂ 0.596 (sd 0.071), mean σ̂_t
  0.808, median σ̂ 0.510.
- Invariances: row order; AR1 and OU time-origin shift; OU time-unit change
  (λ scales by 1/24, nll equal to 1e-9); OU on integer occasions equals AR1
  with φ = e^{−λ}; φ = 0 reduces to `N(Xβ, (σ² + σ_t²) I)`.

## What this does NOT cover

- No drmTMB parity cells yet (the drmTMB side is being rebased); fixtures and
  the drmTMB calls are ready in `test/fixtures/temporal/`.
- Inference: DRModels reports Wald SEs from the observed Hessian for every
  coordinate, as for its other routes. drmTMB deliberately exposes only AR1
  mean-coefficient Wald intervals and no variance/persistence intervals because
  their calibration is not established; no calibration claim is made here
  either.
- `fitted()` / `simulate()` are population-level (temporal effect zero), the
  DRModels convention for structured markers; drmTMB's `fitted()` is
  conditional. `ranef(fit)[:id]` holds the conditional temporal effects in
  data-row order.
- Julia spelling is positional (`temporal(1 | id, occ, ar1)`) because
  StatsModels' `@formula` rejects keyword arguments and string literals.
- Forecasting/newdata for the temporal effect, other families, `sigma`-side
  terms, slopes, REML, weights, missing responses, and combinations with other
  structured terms are refused.

## Neighbours

Every `_split_ranef` caller now either opts in or refuses; read-only callers
that only need the fixed part of a Gaussian formula (`predict`, bridge label
rendering) opt in. The remaining Gaussian helpers that call `_split_ranef`
without the flag (the Laplace validator, the bootstrap simulator in
`src/inference.jl`) refuse a temporal formula, which is the intended boundary.
