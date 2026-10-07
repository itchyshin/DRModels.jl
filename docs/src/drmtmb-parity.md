# Capability parity with drmTMB

DRM.jl is the Julia twin of R's `drmTMB` — a from-scratch distributional-regression
engine (mean μ, scale **σ**, residual correlation **ρ12**) built around a sparse,
exact-gradient Laplace substrate. This page says where DRM.jl stands against
`drmTMB`, and, separately, where the R↔Julia bridge (`engine = "julia"`)
stands against DRM.jl's own direct-Julia engine. **ML is the default** (REML likelihoods aren't comparable
across fixed-effect structures); REML is opt-in.

Naming is kept stable across both packages: `sigma`, never `tau`, for scale;
`rho12` for bivariate residual correlation. Group-level (phylo/spatial/study)
covariance summaries are named structured-effect terms, not `rho12`.

Legend: ✅ available · 🔨 in progress · ⬜ planned · ⚡ DRM.jl advantage.

## The route axis

A capability in `drmTMB`'s R↔Julia ledger can be true in up to **four
different senses**, and conflating them is the single easiest way to
overclaim parity:

| Column | What it answers |
|:-------|:-----------------|
| **Native R** (drmTMB/TMB) | Does the R package's own C++/TMB engine fit this model? |
| **Direct Julia** (DRM.jl) | Does DRM.jl's own engine fit this model, called directly from Julia? |
| **R bridge** (`engine = "julia"`) | Does calling `drmTMB(..., engine = "julia")` from R reach DRM.jl and return a result? |
| **Inference evidence** | What evidence exists that the *numbers* agree (coefficients, SEs, logLik, and — separately, rarely — interval coverage)? |

**"Covered" in the drmTMB ledger is a capability claim, never a coverage
claim.** A row can be `claim_status = covered` — implemented, tested,
documented, with point/SE/logLik parity evidence — while its interval-coverage
fence stays completely untouched. Every row's R-bridge route is `experimental`
by construction (see [Honest gaps](#honest-gaps)): "covered" describes what the
*direct-Julia engine* delivers and what evidence backs it, not that the R
bridge is production-ready.

## Compared routes

Twelve capabilities are tracked, sourced from `drmTMB`'s
`inst/extdata/julia-capabilities.tsv`. The **claim boundary** column carries
each row's evidence honestly — several rows have no native R comparator at
all, so their evidence route is simulation recovery or same-target
native-vs-native agreement, not R-vs-Julia parity in the ordinary sense.

| Capability | Route (plain English) | Native R | Direct Julia | R bridge | Inference evidence |
|:-----------|:-----------------------|:--------:|:-------------:|:--------:|:---------------------|
| `base_gaussian_location_scale` | Gaussian location-scale, `bf(y~x, sigma~z)` | ✅ | ✅ | 🔨 | ✅ coef/SE/logLik parity (2026-08-24/08-15) |
| `biv_gaussian_residual` | Bivariate Gaussian, residual `rho12` | ✅ | ✅ | 🔨 | ✅ one-draw coef/SE/logLik parity, n=400 |
| `gaussian_phylo_mean` | Gaussian phylo random intercept on the mean | ✅ | ✅ | 🔨 | ✅ phylo-SD agreement + measured mean-block coverage; sigma-axis solver defect open |
| `gaussian_response_mask` | Missing Gaussian response, `response="include"` | ✅ | ✅ | 🔨 | ✅ include==drop equality, cross-engine `\|Δlogℓik\| = 4e-10` |
| `biv_q4_phylo_reml` | q4 bivariate phylo location-scale, REML | ✅ | ✅ | ⬜ (halted by design) | ✅ coef parity; measured coverage argues **against** promotion (off-diagonal phylocov under/over-covers) |
| `phylo_count_large_p` | Poisson/NB2 with large-p sparse phylo RE | ✅ | ✅ | 🔨 | ✅ SE parity to p=3000 (199–295×); p=10,000 is DRM.jl-only, no native comparator attempted |
| `phylo_gamma_beta_binomial` | Gamma/Beta/Binomial with phylo RE | ✅ | ✅ | 🔨 | ✅ native-vs-native coef/SE/logLik parity, one fixture |
| `general_covariance_structured` | `relmat()` / relatedness K, Gaussian/Poisson/NB2/Gamma | ✅ | ✅ | 🔨 | ✅ SE parity to 1e-7–1e-6, one seed/family |
| `cross_family_latent` | Shared-latent mixed-family (e.g. Gaussian × Poisson) | ⬜ **no native comparator** | ✅ | 🔨 (via drmTMB's own xfam bridge, not DRM.jl's `bridge.jl`) | ⚡ simulation recovery only — a parity claim is structurally impossible here |
| `engine_control_surface` | `drm_control(optimizer=list(g_tol=…, algorithm=…))` | n/a | ✅ | 🔨 | ⬜ translation/route tests only, no performance/interval claim |
| `plain_binomial_nonphylo` | Fixed-effect binomial trials, no random effects | ✅ | ✅ | 🔨 | ✅ logLik/coef `2.48e-13`, SE `1.27e-9` abs |
| `location_scale_scale` | Location-scale-scale, `sd(group)`/`sd(group, phylogenetic)` | ✅ | ✅ | 🔨 | ✅ exact agreement across the Mizuno M2–M6q ladder (Δlogℓik = 0.000000) |

All twelve rows carry `r_bridge_status = experimental` in the source ledger —
none is promoted past that regardless of how strong the direct-Julia evidence
is. Eleven of twelve rows are `claim_status = covered`; `cross_family_latent`
is `partial` by a **permanent, owner-signed boundary** (D-179 #3) — no native
R comparator for a mixed-family pair can exist in drmTMB, so the row cannot
reach `covered` on the ordinary parity bar and simulation-recovery evidence
is deliberately not being spent on it.

## Performance

- **Verified single-fit speedup:** on the real q4_p100 dataset, same Laplace
  ML marginal, DRM.jl's direct-Julia engine (`fit_q4_sparse_tmb.jl`) converged
  in 1.14 s (logLik −256.51, converged) against drmTMB's 2.48 s (logLik
  −256.52, `converged=false` code 8) — **2.18× faster**, measured 2026-05-30.
  This is a **then**-measured single-machine, single-fixture comparison, not a
  standing warm-workflow claim (see `report/comparison-grid.md`).
- **O(p) sparse phylogenetic path (direct Julia only):** a biological
  per-dimension-variance model scales at k≈1.08 (near-perfect O(p)) from
  p=100 to **p=10,000** (112.9 s), using a Takahashi sparse-precision sampler
  that never forms the dense Σ_phy. A separate paired head-to-head against
  drmTMB 0.6.0 on Totoro found the ratio crosses over near p≈1000 (Julia
  faster at p=100, drmTMB comparable-or-faster from p=1000 up) — an earlier
  "~12× at p=10,000" figure was an **extrapolation** and has been retired.
- **R-bridge warm-workflow performance is NOT yet measured (open gap, tracked
  informally as G5).** No number in this repo compares a *warm* R-bridge call
  against native drmTMB or against DRM.jl's own direct-Julia path.
- A **2026-09-01 q4 bridge fixture run** exists in the repo's evidence trail
  but is **execution evidence only** — it used a mismatched control preset
  and a cold Julia start. It demonstrates the bridge round-trips a q4
  fixture and returns a result; it makes **no bridge speed claim**, and
  should not be read as one.

## Honest gaps

- **Every R-bridge route (`engine = "julia"`) is `experimental`.** All twelve
  ledger rows carry that status regardless of how mature the underlying
  direct-Julia engine is; none has been promoted to a production-ready bridge
  claim.
- **Profile-likelihood and bootstrap parity are unverified.** A raw 343-tip
  diagnostic fixture has shown unbounded upper confidence-interval endpoints;
  this has not been resolved into a calibrated parity or coverage claim.
- **q4 phylogenetic covariance correlations are derived-only.** The scale-axis
  and cross-covariance entries of the q4 PLSM `Sigma_a` block have not been
  through a calibrated coverage campaign; measured evidence on
  `biv_q4_phylo_reml` shows diagonal mean-block entries near nominal but
  off-diagonal phylocov entries systematically miscalibrated (some
  under-covering, some uninformatively wide).
- **No calibrated q4 profile/bootstrap coverage exists.** Point/SE/logLik
  parity is measured on several q4-adjacent rows; interval coverage for
  profile or bootstrap CIs on the q4 engine is not.
- **Package registration waits for v0.7.1 (D-183); CRAN is a separate,
  independently held gate.** Neither this page nor any row on it implies
  registration or CRAN release timing — those are tracked and gated
  elsewhere and are not claims this page makes.

---

Sources: `drmTMB`'s `inst/extdata/julia-capabilities.tsv` (12-row ledger,
`r_bridge_status`/`drmjl_status`/`claim_status`/`claim_boundary` columns);
this repo's `docs/design/capability-status.md` (direct-Julia
implemented/rejected/planned/missing audit); this repo's
`report/comparison-grid.md` (2026-05-30 engine × model × objective grid, the
source of the 2.18× and O(p) figures). Compiled 2026-09-01.
