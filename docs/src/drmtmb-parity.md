# Capability parity with drmTMB

DRModels.jl is the Julia twin of R's `drmTMB`, a from-scratch distributional-regression
engine (mean μ, scale **σ**, residual correlation **ρ12**) built around a sparse,
exact-gradient Laplace substrate. This page says where DRModels.jl stands against
`drmTMB`, and, separately, where the R↔Julia bridge (`engine = "julia"`)
stands against DRModels.jl's own direct-Julia engine. **ML is the default** (REML likelihoods aren't comparable
across fixed-effect structures); REML is opt-in.

Naming is kept stable across both packages: `sigma`, never `tau`, for scale;
`rho12` for bivariate residual correlation. Group-level (phylo/spatial/study)
covariance summaries are named structured-effect terms, not `rho12`.

Legend: ✅ available · 🔨 in progress · ⬜ planned · ⚡ DRModels.jl advantage.

## The route axis

A capability in `drmTMB`'s R↔Julia ledger can be true in up to **four
different senses**, and conflating them is the single easiest way to
overclaim parity:

| Column | What it answers |
|:-------|:-----------------|
| **Native R** (drmTMB/TMB) | Does the R package's own C++/TMB engine fit this model? |
| **Direct Julia** (DRModels.jl) | Does DRModels.jl's own engine fit this model, called directly from Julia? |
| **R bridge** (`engine = "julia"`) | Does calling `drmTMB(..., engine = "julia")` from R reach DRModels.jl and return a result? |
| **Inference evidence** | What evidence exists that the *numbers* agree (coefficients, SEs, logLik, and separately assessed interval coverage)? |

**"Covered" in the drmTMB ledger is a capability claim, never a coverage
claim.** A row can be `claim_status = covered` (implemented, tested,
documented, with point/SE/logLik parity evidence) while its interval-coverage
fence stays separate. The R bridge has its own admission status and restrictions;
strong direct-Julia evidence does not by itself qualify an R bridge route.

## Compared routes

The twelve selected capabilities below are sourced from `drmTMB`'s
`inst/extdata/julia-capabilities.tsv`. The **claim boundary** column carries
each row's evidence; several rows have no native R comparator at
all, so their evidence route is simulation recovery or same-target
native-vs-native agreement, not R-vs-Julia parity in the ordinary sense.

| Capability | Route (plain English) | Native R | Direct Julia | R bridge | Inference evidence |
|:-----------|:-----------------------|:--------:|:-------------:|:--------:|:---------------------|
| `base_gaussian_location_scale` | Gaussian location-scale, `bf(y~x, sigma~z)` | ✅ | ✅ | ✅ supported | ✅ coef/SE/logLik parity (2026-08-24/08-15) |
| `biv_gaussian_residual` | Bivariate Gaussian, residual `rho12` | ✅ | ✅ | ✅ supported | ✅ one-draw coef/SE/logLik parity, n=400 |
| `gaussian_phylo_mean` | Gaussian phylo random intercept on the mean | ✅ | ✅ | 🔨 | ✅ phylo-SD agreement + measured mean-block coverage; sigma-axis solver defect open |
| `gaussian_response_mask` | Missing Gaussian response, `response="include"` | ✅ | ✅ | ✅ supported | ✅ include==drop equality, cross-engine `\|Δlogℓik\| = 4e-10` |
| `biv_q4_phylo_reml` | q4 bivariate phylo location-scale, REML | ✅ | ✅ | 🔨 experimental | ✅ coef parity; measured coverage argues **against** promotion (off-diagonal phylocov under/over-covers) |
| `phylo_count_large_p` | Poisson/NB2 with large-p sparse phylo RE | ✅ | ✅ | 🔨 | ✅ SE parity to p=3000 (199–295×); p=10,000 is DRModels.jl-only, no native comparator attempted |
| `phylo_gamma_beta_binomial` | Gamma/Beta/Binomial with phylo RE | ✅ | ✅ | 🔨 | ✅ native-vs-native coef/SE/logLik parity, one fixture |
| `general_covariance_structured` | `relmat()` / relatedness K, Gaussian/Poisson/NB2/Gamma | ✅ | ✅ | 🔨 | ✅ SE parity to 1e-7–1e-6, one seed/family |
| `cross_family_latent` | Shared-latent mixed-family (e.g. Gaussian × Poisson) | ⬜ **no native comparator** | ✅ | 🔨 (via drmTMB's own xfam bridge, not DRModels.jl's `bridge.jl`) | ⚡ simulation recovery only; a parity claim is structurally impossible here |
| `engine_control_surface` | `drm_control(optimizer=list(g_tol=…, algorithm=…))` | `DRMControl(optimizer=:default/:lbfgs/:bfgs)` on qualified primary ML routes | ✅ | 🔨 | 🟡 Julia route-specific selector tested; R control-surface translation, profile/bootstrap forwarding and other routes remain open. No performance/interval claim. |
| `plain_binomial_nonphylo` | Fixed-effect binomial trials, no random effects | ✅ | ✅ | ✅ supported | ✅ logLik/coef `2.48e-13`, SE `1.27e-9` abs |
| `location_scale_scale` | Location-scale-scale, `sd(group)`/`sd(group, phylogenetic)` | ✅ | ✅ | 🔨 | ✅ exact agreement across the Mizuno M2–M6q ladder (Δlogℓik = 0.000000) |

For these twelve selected rows, the source ledger records four R bridge routes
as `supported` and eight as `experimental`. Ten have `claim_status = covered`;
`cross_family_latent` is `partial`, and `engine_control_surface` is
`experimental`. These statuses apply to each documented route and its limits,
not every model in a family. The ledger also contains other capabilities.
The R bridge remains Experimental as an overall interface; a `supported`
route entry does not make every bridge input stable or supported.
A mixed-family pair without a native R comparator cannot establish ordinary
R-versus-Julia parity through simulation recovery alone.

## Default marginal approximation

drmTMB's native engine integrates every random effect with TMB's Laplace
approximation. DRModels.jl's default, `marginal = :LA`, means "this route's own
integrator", which is not always the Laplace approximation. On the main routes:

- **An ordinary random intercept `(1 | g)` on a non-Gaussian mean** (Poisson,
  Binomial, NegBinomial2, Gamma, Beta, BetaBinomial, Student): per-group
  adaptive Gauss–Hermite quadrature with **5 nodes per group**, centred on each
  group's posterior mode. Tweedie uses 21 nodes and CumulativeLogit 41;
  LogNormal's single node is exact.
- **A correlated `(1 + x | g)` on a non-Gaussian mean**: the same adaptive
  quadrature, 5 nodes per axis.
- **Gaussian `sigma ~ 1 + (1 | g)` and ZeroOneBeta `(1 | g)`**: a fixed
  32-node Gauss–Hermite grid.
- **Crossed intercepts `(1 | g) + (1 | h)`, and `phylo` / `relmat` / `animal` /
  `spatial` terms, on a non-Gaussian mean**: the Laplace approximation, as in
  drmTMB (LogNormal's are exact, being Gaussian fits on log y).
- **Gaussian random or structured effects on the mean, with no random effect on
  `sigma`**: the exact marginal, which the Laplace approximation also gives
  exactly. With the same `(1 | g)` on both the mean and `sigma`, the default is
  the Laplace approximation.

On the adaptive-quadrature cells the two packages fit the same model with
different integrators, so their log-likelihoods and estimates differ by more
than rounding. On seven test datasets (Poisson, Beta, NegBinomial2, Gamma and
Student random intercepts) the default and drmTMB log-likelihoods differed by
0.01 to 0.24 units and the estimates by up to about 3%. In every case drmTMB's
value was the one further from a 61-node quadrature reference; the default
stayed within about 5e-4 of it. A later Poisson `(1 | g)` comparison
(2026-10-15, three simulated datasets) showed a smaller gap, about 0.0014.

- **To match drmTMB, pass `marginal = :Laplace`.** On ten test datasets
  (Binomial included) the refitted log-likelihoods agreed with drmTMB's to
  within 5e-11 and the estimates to within 5.4e-8 (relative); in the 2026-10-15
  Poisson comparison, the dataset checked agreed to 5.6e-9 in log-likelihood
  and 1e-11 in the estimates. `:Laplace` is accepted for one ordinary `(1 | g)`
  on the mean of Poisson, Binomial, NegBinomial2, Gamma and Beta (with
  `sigma ~ 1`) or Student, and for a Gaussian random intercept on `sigma`
  (`sigma ~ 1 + (1 | g)`, fixed-effect `sigma` covariates allowed) with a
  fixed-effect mean, by maximum likelihood. `LogNormal()` also accepts
  `:Laplace` for one ordinary mean random intercept: it returns the same exact
  single-node mode-centred quadrature fit as its default and records the
  requested label. Other LogNormal models refuse this selector. For families
  without a documented `:Laplace` route, use their default; a `marginal`
  keyword is not accepted by every family. Crossed and structured
  non-Gaussian routes generally use Laplace by default, with exact Gaussian
  calculations for the delegated LogNormal routes.
- **On the adaptive-quadrature cells, `:Laplace` is for parity, not a better
  default.** The Laplace approximation is adaptive quadrature with a single
  node, so the default's extra nodes usually make it the more accurate integral
  of the same model, as on the datasets above. Laplace loses accuracy when
  groups are small or counts sit near zero (binary data above all), where each
  group's posterior is far from Gaussian. Gaussian `sigma ~ 1 + (1 | g)` is the
  exception: its fixed grid loses accuracy as groups grow, so for large groups
  `:Laplace` (or `:AGHQ`) is the more accurate choice; see the scale-axis row
  of [What is tested](capabilities.md).
- **Compare like with like.** Log-likelihoods, likelihood-ratio tests, AIC and
  BIC are comparable only between fits that use the same approximation, and
  across the two packages only when both use the Laplace approximation.
  `fit.marginal` records the option (`:LA`, `:Laplace`, `:AGHQ` or `:VA`; the
  default Gaussian fit with `(1 | g)` on both the mean and `sigma` is tagged
  `:Laplace`). `lrtest` and `anova` refuse a pair whose tags differ (a fit with
  no random effects, whose log-likelihood is exact, may be paired with any
  non-variational fit). `aic`, `bic` and `aicc` refuse a `:VA` fit but, taking
  one fit at a time, cannot check that two fits match. The tag is not the
  integrator: two default fits on different routes, such as `(1 | g)` against
  `(1 | g) + (1 | h)`, use different integrators, and no check catches that.

The [Laplace vs variational marginals](model-guides/marginal-la-vs-va.md) guide
covers the approximations in more depth.

## Performance

- **Retained single-fit comparison:** on the real q4_p100 dataset, using the
  same Laplace ML objective, DRModels.jl's direct-Julia engine
  (`fit_q4_sparse_tmb.jl`) converged in 1.14 s (logLik −256.51), while drmTMB
  took 2.48 s (logLik −256.52, `converged=false` code 8). The R fit's false
  convergence limits interpretation of the timing ratio. These measurements
  describe one machine and fixture on 2026-05-30; they do not establish current
  warm-workflow performance (see `report/comparison-grid.md`).
- **O(p) sparse phylogenetic path (direct Julia only):** a biological
  per-dimension-variance model scales at k≈1.08 (near-perfect O(p)) from
  p=100 to **p=10,000** (112.9 s), using a Takahashi sparse-precision sampler
  that never forms the dense Σ_phy. A separate paired head-to-head against
  drmTMB 0.6.0 on Totoro found the ratio crosses over near p≈1000 (Julia
  faster at p=100, drmTMB comparable-or-faster from p=1000 up). An earlier
  "~12× at p=10,000" figure was an **extrapolation** and has been retired.
- **R-bridge warm-workflow performance is NOT yet measured (open gap, tracked
  informally as G5).** No number in this repo compares a *warm* R-bridge call
  against native drmTMB or against DRModels.jl's own direct-Julia path.
- A **2026-09-01 q4 bridge fixture run** exists in the repo's evidence trail
  but is **execution evidence only**; it used a mismatched control preset
  and a cold Julia start. It demonstrates the bridge round-trips a q4
  fixture and returns a result; it makes **no bridge speed claim**, and
  should not be read as one.

## Honest gaps

- **Every R-bridge route (`engine = "julia"`) is `experimental`.** All twelve
  ledger rows carry that status regardless of how mature the underlying
  direct-Julia engine is; none has been promoted to a production-ready bridge
  claim.
- **Profile-likelihood and bootstrap parity are unverified.** This page does
  not establish cross-engine agreement or interval coverage for either method.
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
  registration or CRAN release timing, which are tracked and gated
  elsewhere and are not claims this page makes.

---

Sources: `drmTMB`'s `inst/extdata/julia-capabilities.tsv` (12-row ledger,
`r_bridge_status`/`drmjl_status`/`claim_status`/`claim_boundary` columns);
this repo's `docs/design/capability-status.md` (direct-Julia
implemented/rejected/planned/missing audit); this repo's
`report/comparison-grid.md` (2026-05-30 engine × model × objective grid, the
source of the retained single-fit and scaling figures). Compiled 2026-09-01. The default marginal
approximation section was added 2026-10-16. Its seven- and ten-dataset figures
come from this repository's class-1 Laplace parity comparison (2026-09-30,
Poisson, Binomial, Beta, NegBinomial2, Gamma and Student random intercepts);
its 2026-10-15 Poisson figures from a comparison of drmTMB at commit `ca3ddef`
with DRModels.jl at commit `f71b292` (#1070).
