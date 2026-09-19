# API stability

*The stability promise of the v0.7 line, stated precisely. Machine-checked by the package API-stability gate, which
classifies **every** exported name into exactly one tier and fails if a new export appears
unclassified, a stable name vanishes, or a tier changes without a reviewed edit.*

## The promise

*Versioning follows the R twin's parity level. Julia General registration remains
deferred; a formal SemVer `1.0` will be coordinated with the twin. Julia's 0.x
convention allows minor-version breakage; this page plus the test gate are the
promise that we will not use that allowance on the Stable tier.*

From `v0.7.0`, names in the **Stable** tier — the `bf()`/`drm()` grammar, the fourteen family
constructors, the structured-effect markers, and the accessor/inference/plotting surface — keep
their names, meanings, and conventions across the `0.7.x` line and beyond. The conventions are drmTMB's: scale is
**`sigma`** (never `tau`), the bivariate residual correlation is **`rho12`**, meta-analysis is
`gaussian()` + **`meta_V`**. Breaking any of these is a co-versioned major event decided with the twin, never a quiet 0.x bump.

## The three tiers

**Stable** — the promise above. The stable API contract is the authoritative
list and is checked against the implementation.

**Experimental — exported, usable, exempt.** These work today and are tested, but their shape may
change between releases, and each carries its reason:

- **`r2_constant_sigma`** — the number is settled (it is the ordinary `lm()` R², and equals it
  exactly on a constant-σ Gaussian fit), but the REFUSALS are the part a caller programs against,
  and they may widen. In particular a *marginal* or *conditional* R² for random-effect fits
  (Nakagawa & Schielzeth) would reshape this surface, and that decision has not been taken;
- the **R bridge** (`drm_bridge`, `drm_bridge_inference`, `drm_listwise`) — it is experimental,
  so its accepted models and returned details may change;
- the **cross-family surface** (`mf_*`, `associate_pairs`, `latent_normal`, `association`,
  `PairAssociation`, `integration_diagnostics`) — it has a deliberately narrow documented scope;
- the **penalized-MAP surface** (`drm_phylo_penalty*`, `PhyloPenalty`,
  `PhyloCorPenaltyNeedsTwoSD`) and **bivariate meta** (`meta_vcov_bivariate`) — newer surfaces
  whose ergonomics are still settling;
- the **VA/ELBO marginal** (`marginal = :VA`) — reachable through stable `drm()` but
  Experimental-labelled on its own pages; its behaviour and coverage may change between releases;
- the **prepared joint missing-predictor surface** (`PreparedJointModel`,
  `prepared_joint_model`, `fit_prepared_joint`, `mi`, and related result and
  summary types) — implemented and tested, while formula and bridge ergonomics
  continue to settle.

**Engine** — the computational spine (`AugProblem`, `make_problem`, `fit_q4_sparse_tmb`,
`estep_mode`, the `coevo_*` and `fz_*` families, tree utilities, packers). Exported for advanced
scripts and benchmarks; stable in practice but **not** part of the promise, the same way a
language's internals are not.

## What the promise does and does not cover

It covers names, argument meanings, and return conventions on the Stable tier. It does **not**
cover: numerical trajectories (an optimiser or tolerance improvement may move estimates within
documented accuracy), the Experimental and Engine tiers, or interval *coverage* — interval claims
throughout this package are **capability parity, not coverage**; the ledger's `coverage_claimed`
fences are permanent documented boundaries.

## Deliberate exclusions at v0.7.0

- **Bivariate Student structured markers** (`phylo`/`relmat`/`animal`/`spatial` on
  `Student()` bivariate) are **outside** the frozen surface — matching
  drmTMB, whose own `biv_student()` defers structured effects. Bivariate **LogNormal** structured
  markers are inside: they delegate to the exact/q4 Gaussian engines on `log(y)` and are tested.
- The prepared joint missing-predictor API is available as an Experimental
  post-v0.7 surface. Its inclusion here does not widen the Stable-tier promise.
