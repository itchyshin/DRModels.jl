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
- Likelihood: against a BigFloat dense oracle, worst error 1.4e-15 relative
  (K = 3, 6, 12; PACs alternating at ±0.995). A Float64 dense oracle loses
  about 1e-7 there; that is the oracle's own conditioning, so it was replaced.
- Parity with drmTMB #1449 (final head 90c740791): logLik ≤ 9.6e-12 on all three
  cells, lag correlations ≤ 4.9e-10. drmTMB's printed article values are
  reproduced, including the tmbprofile interval for mu:treatment
  [0.2886, 0.4354] (endpoints within 1.3e-6).
- Recovery (400 × 6): lag correlations within 0.06, σ 0.906 (truth 0.9).
- Full suite on 1.10.12: 4/4 shards pass (1.13: see the PR). Documenter
  build EXIT=0.

## What this does NOT cover

- No calibration claim of its own. drmTMB's 4,000-fit campaign qualifies
  mean-effect profiles in its primary panels only, and that result is quoted
  as drmTMB's.
- Wald SEs print for every coordinate (DRModels convention); drmTMB withholds
  them.
- Missing responses are refused (drmTMB re-checks the retained panel after
  omission).
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
