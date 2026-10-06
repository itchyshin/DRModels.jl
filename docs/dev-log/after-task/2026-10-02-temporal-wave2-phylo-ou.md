# After-task: temporal wave 2A — phylogenetic stable intercept + OU (2026-10-02)

## Scope

Owner decision D-311, slice A: the Julia twin of drmTMB's paired provider
`phylo(1 | species, tree = tree) + temporal(1 | species, time = elapsed,
structure = "ou")` (drmTMB draft PR #1448, `claude/temporal-phylo-ou-land`,
rescued from `codex/phylo-temporal-ou-exec-v1-20260909`). Branch
`claude/temporal-phylo-ou` from origin/main 51ea90697; PR #917 (draft).
Semantics were read, not copied, from drmTMB's `R/temporal.R`, the
reference-helper tests and the article. drmTMB is GPL, and no drmTMB source is
in this PR. Only generated numbers and the article's generated data are
included.

## What changed

- `src/temporal.jl`: paired route in `_drm_gaussian_temporal`, with drmTMB's
  refusals: OU only; an unlabelled intercept-only `phylo()` on the same
  grouping; no `(1 | species)`; ≥ 3 species, each with ≥ 2 times; ≥ 3
  distinct lags; tree tips exactly the observed species; a tree is required.
  New functions: the tree plan, the upward pruning pass, the downward mode
  pass, and the fresh-field draw. `_fit_temporal_gaussian(…; phylo)`,
  `temporal_parameters(…).sd_phylo`, and `simulate` with a fresh phylogenetic
  vector.
- `src/gaussian_core.jl`: the router now receives `structured_slope` and
  `tree`.
- Tests `test/test_temporal_phylo_ou.jl` (133); fixtures
  `phylo_ou_species.*` (simulated) and `vignette_phylo_ou_species.*`
  (drmTMB's article data); generator `test/parity/gen_temporal_wave2_parity.R`;
  parity cells `phylo-ou-species` and `vignette-phylo-ou`. The parity helpers
  gained tree files, `sd_phylo`, and an optional `[tol].boundary`.
- Tutorial section, capabilities row, news fragment.

## Design decision

The exact marginal mixes a tree covariance with block-diagonal OU blocks for
each species. Each series is one species, so it depends on the stable field
only through `a_s 1`. Wave 1's filter `with_one` sums (`c_s`, `d_s`) turn each
species into a Gaussian leaf factor. The tree is then integrated by an upward
pruning pass, which gives `log1p(ℓα)` and `ℓβ²/(1+ℓα)` terms. This is the
zero-fill sparse Cholesky of the joint precision, written as scalar
recursions: O(n), and it works for any number type (ForwardDiff). Woodbury
over the phylo part with CHOLMOD was rejected because CHOLMOD cannot take
Dual numbers. A dense K×K species-level solve was rejected because it is
O(m³). The tip-correlation scale (drmTMB, and DRModels' closed-form
`phylo(1|g)`) is exact through the leaf scaling `1/√h_s`, ultrametric or not.

## Evidence (re-measured at the final head; Julia version named per figure)

- Dense oracle (tree covariance built independently from shared branch
  lengths), Julia 1.10.12: worst |Δnll| 9.2e-13 over random θ on ultrametric
  trees and at decay e^6 / e^-8 / e^-20, σ_a e^-15 / e^3 and σ_t e^-15.
  Zero-length internal and tip branches: dense oracle to 1e-10. The pruning
  pass alone on non-ultrametric trees matches the dense identities to 1e-10.
  The modes match the dense formulas to 1e-8.
- Parity with drmTMB #1448 final head 012258e9f, Julia 1.10.12. I regenerated
  the cells there and diffed `expected.toml`: identical to 66ce5750d.
  - `phylo-ou-species`: logLik 1.1e-12, estimates ≤ 2.1e-11 relative,
    conditional fitted values 2.3e-12.
  - `vignette-phylo-ou` (60 species): logLik 5.5e-12, estimates ≤ 2.6e-12,
    fitted values 1.7e-12.
  - The profile endpoints in the article differ by 1.2e-6 (root-search
    tolerance).
- Recovery smoke (m = 150, Julia 1.10.12): σ_a 0.48 (truth 0.6), σ_t 0.84
  (0.75), λ 0.46 (0.45), σ 0.35 (0.4), slope 0.488 (0.45).
- Full suite: see the PR comments for the final-head shard runs.

## Fixed during the work

- The first simulated fixture put σ_a at zero (seed luck). It now uses 24
  species with a larger stable SD, and drmTMB's fit is interior.
- A `[tol].boundary` loosening for boundary cells was added for drmTMB's
  earlier 8-species article data. Once the article moved to 60 species no
  cell used it, so it was removed.

## What this does NOT cover

- No interval or calibration claim (drmTMB marks this model development-only).
- `fitted`/`predict` are population-level (drmTMB: conditional).
- Missing responses are refused (drmTMB re-checks rows retained after
  omission). The bridge relies on drmTMB's R side to strip `tree = tree`.
- A separable phylogeny × time field is a different model and is not
  implemented.
- The parity cells must be regenerated when #1448 changes or merges.
- The σ boundary rule compares σ̂ with the marginal sd(y), as drmTMB does, so
  it can fire falsely when a covariate dominates the response (for example
  y = 2000x + OU). This is kept for parity and raised upstream.
- A branch length missing from the Newick reads as 0 (drmTMB, via ape,
  would refuse a tree with no branch lengths).
- `profile_curve` on the paired fit does not warn (only `confint(:profile)`
  does, as drmTMB).

## Follow-up after drmTMB review (#1447 07d1612ea, #1448 012258e9f)

- Temporal boundary diagnostic (`_temporal_boundary`, `check_drm(fit).temporal_boundary`,
  fit-time warning through `_warn_variance_boundary`): drmTMB's four rules,
  for wave-1 AR1/OU and the paired fit. Test file `test/test_temporal_boundary.jl`
  covers each threshold on both sides and drmTMB's own boundary panel. On
  that panel DRModels.jl reaches decay 1.4e-18 with drmTMB's β, σ and SD (to
  1e-9). At drmTMB's θ̂ the exact (256-bit dense) logLik is −45.5124249517,
  while drmTMB prints −45.5124224656: a 2.5e-6 loss in drmTMB's objective as
  decay → 0. This is reported to the drmTMB lane.
- I tried and dropped an OU "boundary polish" restart. On simulated
  random-intercept panels the decay MLE is interior (≈ 0.006, profile checked),
  and on drmTMB's panel the plain optimiser already reaches the boundary.
- Profile `confint` on the paired fit warns that the intervals are
  uncalibrated (drmTMB #1448). The unobserved-tips refusal now names the tips.
- Parity cells regenerated from 012258e9f: the numbers are unchanged and
  only the provenance is new.

## Follow-up after the #917 review (2026-10-02)

- Non-ultrametric trees are refused on the user route with drmTMB's relative
  tolerance and wording. The likelihood is still checked on such trees
  internally (`require_ultrametric = false`).
- Zero-length branches are accepted. `_temporal_phylo_tree` reads the Newick
  with the shared `_parse_node!` reader, but not with `augmented_phy`'s
  sparse-precision assembly, which stays unchanged and still refuses 0.
- The boundary diagnostic gains drmTMB's random-effect SD rule (smallest SD
  below 1e-4). A true σ_a = 0 panel (12 species × 5 times) gives sd_phylo
  3.9e-10 and is flagged.
- The `[tol].boundary` mechanism is removed. `:homtoep` references moved to
  #918. Licence comments mark the refusal and boundary wording as mirroring
  drmTMB.
