# After-task: temporal wave 2B — homogeneous Toeplitz (2026-10-02)

## Scope

Owner decision D-311, slice B: the twin of drmTMB's
`temporal(1 | id, time = occ, structure = "homtoep")`. drmTMB draft PR #1449
(`claude/temporal-homtoep-land`, rescued from
`codex/temporal-homtoep-v1-20260910`) contains #1448. Branch
`claude/temporal-homtoep`, stacked on `claude/temporal-phylo-ou` (#917),
because both slices extend the same router, layout, accessors, parity
helpers, generator and article. PR #918 (draft). Semantics were read, not
copied; no drmTMB source is included.

## What changed

- `src/temporal.jl`:
  - The parser accepts `homtoep`.
  - Layout enforces drmTMB's panel rules, each with drmTMB's message:
    integer occasions, 3–12 common occasions, equal spacing, complete
    schedule per series (incomplete ones named).
  - An ordinary `(1 | id)` is refused.
  - New functions: Durbin–Levinson (`_homtoep_levinson`), lag correlations,
    the per-series innovations likelihood, `_fit_temporal_homtoep` (θ = β,
    log σ, atanh PACs; no latent states), and `_homtoep_simulate`.
  - `temporal_parameters` gains `cor` and `pac`.
- `src/bridge.jl` accepts `structure = "homtoep"`. `src/introspection.jl`
  and `src/summary.jl` handle `:temporal_pac` (atanh scale).
- Tests: `test/test_temporal_homtoep.jl` (97).
- Fixtures: `homtoep_panel6.csv` and `homtoep_neg4.csv` (simulated; the
  generator writes its own Levinson step), plus `vignette_homtoep_sites.csv`
  (drmTMB's article data).
- Three parity cells. The generator takes `phylo | homtoep | all`.
- Tutorial section (the page is now *Temporal AR1, OU and Toeplitz effects*),
  capabilities row, news fragment.

## Design decision

σ is the total within-series SD, and there is no separate process SD,
residual SD or `(1 | id)`: drmTMB's identifiability decision, mirrored
exactly. R is parameterised by its partial autocorrelations, which cover
every positive-definite Toeplitz correlation matrix and nothing else. The
likelihood is the Levinson innovations factorisation, with
`log(1 − tanh²κ) = −2 log cosh κ` written in stable form. A K×K Cholesky
(K ≤ 12) would also be exact. Levinson avoids forming R, keeps the
cancellation-free log variances, and gives `simulate` the innovations form
directly.

## Evidence

- Parameterisation: the PACs of the BigFloat R, recomputed by dense Schur
  complements, equal tanh κ to 1e-40. While building this check I found a
  Float64 `log(2.0)` constant that capped the BigFloat path near 1e-16, which
  is invisible in Float64 but amplified by near-singular R. I replaced it
  with a type-generic constant.
- Likelihood: against a BigFloat dense oracle, worst error 3.7e-15 relative
  (≤ 4e-15; K = 3, 6, 12; PACs alternating at ±0.995; Julia 1.10.12). A Float64 dense oracle loses
  about 1e-7 there; that is the oracle's own conditioning, so it was replaced.
- Parity with drmTMB #1449 (final head 90c740791): logLik ≤ 9.6e-12 on all three
  cells, lag correlations ≤ 4.9e-10. drmTMB's printed article values are
  reproduced, including the tmbprofile interval for mu:treatment
  [0.2886, 0.4354] (endpoints within 1.3e-6).
- Recovery (400 × 6): lag correlations within 0.06, σ 0.906 (truth 0.9).
- `test/test_temporal_homtoep.jl`: 163 tests (was 111 before the #918
  review batch). Full suite: see the PR comments for the final-head shards.

## What this does NOT cover

- No calibration claim of its own. drmTMB's 4,000-fit campaign qualifies
  mean-effect profiles in its primary panels only, and that result is quoted
  as drmTMB's.
- The `1e18` failure sentinel of the objective is kept, though no admissible
  θ reaches it: at ±40 the true objective is about 1e173 and finite.
- There is no boundary rule for the partial autocorrelations; drmTMB has none
  for homtoep either. The temporal boundary diagnostic covers homtoep through
  the total-σ rule only.
- `bootstrap_ci` still returns percentile intervals for σ and the PACs, and
  `profile_curve` still draws non-mean coordinates. Neither is a confint
  route that drmTMB refuses, but the "intervals deferred" caution applies.
- Neighbour finding, not fixed here: wave-1 AR1/OU `residuals(fit; type =
  :quantile)` standardises by σ only, ignoring the temporal (and `(1 | id)`)
  covariance. drmTMB's Pearson residuals there are conditional on the modes.
- Heterogeneous Toeplitz, unstructured covariance, and homtoep combined with
  `phylo()` are not implemented (drmTMB pairs only OU).
- The parity cells must be regenerated when #1449 changes or merges.

## Follow-up after drmTMB review (#1449 final head 90c740791)

- New refusal, drmTMB's estimability floor: at least as many series as
  occasions. It is checked after the complete-panel rule, as in drmTMB, and
  tested at S = K − 1 (refused) and S = K (fits).
- drmTMB's stability points: atanh PACs (8, −8, 5, 0, 3),
  (15, 15, −15, 12, 0.1), (18.5, 0, …) and ±40. The objective is finite and
  matches an independent 2048-bit Yule–Walker prediction-error reference to
  ≤ 2.8e-14 relative, even where `tanh` rounds to ±1. The reference solves
  `R φ = r` by dense LU at every order. The true objective there reaches
  1e173, because the innovation variances fall to about 1e-170.
- Parity cells regenerated from 90c740791. drmTMB's likelihood rewrite moved
  its own numbers by up to 1e-13. DRModels.jl matches logLik to ≤ 9.6e-12 and
  the lag correlations to ≤ 4.9e-10 (abs). The article data are identical.
- The temporal boundary diagnostic from #917 also covers `homtoep`, through
  the total-σ rule only.

## Follow-up after the #918 review (2026-10-02)

- Inference scope mirrors drmTMB (`validate_temporal_wald_parm`,
  `vcov.drmTMB`, `validate_temporal_profile_parm`):
  - `vcov`, `stderror`, Wald `confint` and `predict(se = true)` are refused;
  - profiles cover the mean coefficients only (the default selects `:mu`);
    `:sigma` and `:temporal_pac` are refused;
  - `profile_targets` marks the non-mean rows not ready; `coeftable` and
    `show` print NaN SEs; the bridge ships a NaN vcov.
- Whitened residuals (`type = :quantile`): L⁻¹ r per series, matched to a
  dense Cholesky to 1e-10 and to drmTMB's Pearson residuals (new
  `[residuals]` parity block) to 1e-6.
- Missing responses as drmTMB, verified on Totoro against 90c740791:
  - one NA row is refused as an incomplete series;
  - an all-NA last occasion fits with K − 1, logLik −213.499788672539,
    matched to 1e-8;
  - an all-NA middle occasion is refused as not equally spaced;
  - duplicate keys are still refused when the duplicate's response is NA.
- Parity tests enforce drmTMB's profile endpoints at 1e-5. The measured gap
  is 1.3e-6 on vignette-homtoep and at most 6.6e-6 over all six profile cells
  (homtoep-neg4, vignette-ou-ri), so 5e-6 would have been too tight for the
  wave-1 and neg4 cells.
- The `(1 | id)` refusal uses drmTMB's group-free wording.
