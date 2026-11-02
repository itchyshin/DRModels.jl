# DRModels.jl capability status (R <-> Julia parity view)

This file is the Julia-side input to the mission-control R <-> Julia parity
board. It uses the **same model-level capability names** as drmTMB's
`docs/design/capability-status.md` so the mission-control server can match
rows by name across the two twins.

Status words:

- `implemented` -- found real code AND a test file exercising it (an export in
  `src/DRModels.jl`'s export list plus a matching `test/test_*.jl`, and for the
  default-suite claims, included in `test/runtests.jl`).
- `rejected` -- found real code that deliberately refuses the capability (an
  explicit `throw(ArgumentError(...))` or equivalent guard), verified by
  reading the guard.
- `planned` -- a tracked, documented stub (an issue number, an explicit `error()`
  pointing at a GitHub issue, or a `src/experimental/` file not yet wired into
  the module).
- `missing` -- no implementation found anywhere in `src/`, `docs/`, `README.md`,
  `ROADMAP.md`, or `HANDOVER.md`.
- `experimental` -- real code exists, is exported, and has a test file, but a
  recorded decision fences it out of the frozen/parity-claimed surface (e.g.
  a v1.0 boundary) or narrows its scope well below a native capability; not
  `implemented` until that fence is lifted by the same decision process.

Nothing below is marked `implemented` without a source file and a test file
both found by reading the repository. The public capabilities page is a
reader-facing summary; this ledger records the source-and-test basis and the
scope limits behind each status.

## Response families

`src/DRModels.jl`'s export line lists all 13 named families plus the `zi`/`hu`
count modifiers and `zoi`/`coi` beta boundary modifiers; the bivariate
Gaussian route and the q4 phylogenetic bivariate location-scale (PLSM) model
are separate exports. Every row below has a source file under `src/` and a
`test/test_*.jl` file.

| Capability | Status |
|---|---|
| Gaussian location-scale (ML) | implemented |
| Bivariate Gaussian coscale (rho12) | implemented |
| Student-t location-scale | implemented |
| LogNormal location-scale | implemented |
| Gamma location-scale | implemented |
| Poisson counts | implemented |
| NegBinomial2 (NB2) counts | implemented |
| Zero-inflated Poisson (ZIP) | implemented |
| Zero-inflated NB2 (ZINB) | implemented |
| Beta proportions | implemented |
| Truncated NB2 (zero-truncated counts) | implemented |
| Hurdle NB2 | implemented |
| Cumulative logit (ordinal) | implemented |
| Beta-binomial proportions | implemented |
| Zero-one-inflated beta | implemented |
| Tweedie (compound Poisson-Gamma) | implemented |
| Skew-normal location-scale | implemented |
| Binomial (logistic) | implemented |

`Skew-normal location-scale` is marked `implemented` from direct source
evidence (`src/skewnormal.jl`, exported `SkewNormal` in `src/DRModels.jl`,
`test/test_skewnormal.jl`). The current public family table also lists
SkewNormal and correctly limits it to fixed effects.

## Random-effect structure

`src/gaussian_ranef.jl`, `src/gaussian_structured.jl`, and
`src/sparse_laplace_glmm.jl` back the Gaussian and non-Gaussian random-effect
rows; each has a `test/test_*.jl` file cited in `docs/src/capabilities.md`'s
"Random-effect structures" section.

| Capability | Status |
|---|---|
| Gaussian random intercept (mean) | implemented |
| Gaussian random slope (mean) | implemented |
| Gaussian random effect on sigma (scale) | implemented |
| Gaussian phylogenetic random intercept (mean) | implemented |
| Gaussian phylogenetic random intercept + slope, two SDs (mean) | implemented |
| Gaussian spatial random intercept (mean) | implemented |
| Gaussian animal-model random intercept (mean) | implemented |
| Gaussian relmat random intercept (mean) | implemented |
| Non-Gaussian phylogenetic random intercept (mean) | implemented |
| Non-Gaussian phylogenetic location-scale (μ + log σ) | implemented |
| Non-Gaussian ordinary random intercept (TMB-convention Laplace) | implemented |
| Coupled NB2 ordinary location-scale random intercept | implemented |
| Tweedie random intercept (mean) | implemented |

`Non-Gaussian ordinary random intercept (TMB-convention Laplace)` is
`implemented` for the scoped public `marginal = :Laplace` route in
`src/ordinary_laplace.jl`: Poisson, Binomial, NegBinomial2, Gamma, and Beta,
with one unstructured `(1 | g)` term on `mu`; scale families require
`sigma ~ 1`, and the route is ML-only. The source is included by
`src/DRModels.jl`; `test/test_ordinary_laplace.jl` exercises the public route
and its refusals and is picked up by the default test discovery. This row does
not imply support for correlated slopes, structured effects, REML, or
calibrated uncertainty.

`Coupled NB2 ordinary location-scale random intercept` is `implemented` for
the shared correlated `(1 | p | group)` term on both `mu` and `sigma`. The
public NegBinomial2 method routes that formula through
`src/locscale_frontend.jl` to the joint two-dimensional Laplace marginal;
`test/test_locscale_frontend.jl` checks the public fit against the engine,
including coefficients, covariance, standard errors, summary, and Wald
interval output. This is the ordinary iid group route (`Q = I`), distinct
from the structured phylogenetic location-scale row above.

`Gaussian phylogenetic random intercept + slope, two SDs (mean)` is
`implemented` (#620; was `refused` by #621 at the previous pin):
`phylo(1 + x | species)` on the Gaussian mean fits drmTMB's model — two
independent phylogenetic fields (intercept `a ~ N(0, σₐ² C)`, slope
`b ~ N(0, σ_b² C)`, no correlation). In `src/drmTMB.cpp` that is
`model_type == 1` + `has_phylo_mu` with `q_phylo == 2` and both fields on `mu`,
so `has_cross_dpar_phylo` is false and the per-field
`exp(-2·log_sd_phylo(k))·uₖᵀQuₖ` branch runs with no cross term. DRModels.jl fits it
as the closed-form dense marginal `V = D + σₐ² Z C Zᵀ + σ_b² Zₓ C Zₓᵀ`
(`src/gaussian_structured.jl` `_fit_phylo_slope_gaussian`), the same five free
parameters drmTMB optimizes. Measured same target as drmTMB 67703f541 on the
committed fixture in `test/test_phylo_slope_two_sd.jl`: logLik, both SDs, β and
log σ agree to ≤ 6.7e-13.

BOUNDARY (written, not silent; CORRECTED 2026-09-05 after measurement).
drmTMB fits `phylo(1 + x | g)` on Poisson and NegBinomial2 too, and it fits the
**same** independent two-SD model there, not a different one. Measured on
drmTMB main (`engine = "tmb"`, 30-tip `ape::rcoal`, Poisson): the untagged
`phylo(1 + x | site)` returns SD rows `phylo(1 | site)` and
`phylo(0 + x | site)` with `corpars` **empty**, while the tagged
`phylo(1 + x | p | site)` returns `corpars` = `cor(mu:(Intercept),mu:x | p |
site)` = 0.824. The estimated intercept–slope correlation
(`has_phylo_mu_q2_covariance`) therefore belongs to the *tagged* formula, which
is a different construct from the one refused here; drmTMB refuses this formula
outright on Gamma ("intercept-only in this q=1 route"). DRModels.jl still refuses
every non-Gaussian family on this route, for the accurate reason: it is the
EXACT closed-form Gaussian marginal, which does not extend to a non-Gaussian
likelihood — the count families need a two-field Laplace route that does not
exist yet. The prior wording named a target difference that measurement does
not support; the refusal itself is unchanged. `phylo(0 + x | g)`, multi-slope
forms, a second structured component, ordinary and `sigma`-side random effects, a
structured (`phylo`) `sigma`, the `sd(...)`/`sd_phylo(...)` location-scale-scale
submodels, REML, missing responses, the sparse algorithms and the
parametric-bootstrap marginal simulator all stay refused with a named error --
and those refusals are raised next to `_split_ranef`, ABOVE every route that can
return, so no engine written for the intercept-only cell can receive this formula
and quietly fit `phylo(1 | g)` instead. The R-side bridge
receipt (coef labels for the two-row `resd` block, design 258) is the drmTMB
lane's, after this lands.

`Non-Gaussian phylogenetic random intercept (mean)` is `implemented` via the
sparse augmented-state Laplace engine (`src/sparse_laplace_glmm.jl`) for
Poisson, NegBinomial2, Gamma, Binomial, and Beta, each with its own
`test/test_*_phylo_laplace.jl` plus a gradient gate. This is a genuine R <-> Julia
gap: drmTMB reports this row `scope-limited` (mixed rejected/scope-limited
across families).

`cumulative_logit()`'s unlabelled, intercept-only `phylo(1 | species)` on `mu`
(#563 S8 follow-on) reuses the same sparse-Laplace engine — the ordinal
likelihood's cutpoints are the one genuinely new piece, generalising the
scalar/`Xσ`-chained nuisance-parameter IFT gradient (`_phylo_mean_laplace_hetero_fg`,
#164) from a scalar dispersion to the `K-1` cutpoint increments. Same-target
vs drmTMB 0.7.0 (`test/test_cumlogit_phylo.jl`, 60-tip `ape::rcoal` tree,
Brownian phylo intercept): measured gaps of 2.7e-7 (β), 9.6e-7 (cutpoints),
1.6e-6 relative (phylo SD, correlation-scale), 3.8e-5 (logLik) — both engines
land on essentially the same optimum, tighter than the iid slice's GHQ-vs-TMB
comparison because both sides run a Laplace approximation of the same model
here. Cannot be combined with an ordinary `(1 | g)`/`(0 + x | g)` random
effect on `mu` (drmTMB parity); `relmat`/`animal`/`spatial` structured markers
on `mu` stay refused for `cumulative_logit()` (drmTMB scope, not implemented
there either).

`Non-Gaussian phylogenetic location-scale (μ + log σ)` is `implemented`
(`closes #202`): shared structured RE on mean **and** scale via grammar B
`(1 | p | phylo(species))` on both axes, routed through `src/locscale_*.jl`.
Public evidence: `test/test_public_phylo_locscale.jl` (NB2 recovery + Gamma
smoke; dual issue-text `phylo(1|sp)` on both axes still rejected). Private
Gamma recovery + FD ≤1e-6 already in `test/test_phylo_locscale.jl` (#253).
**D-94 honesty:** drmTMB still skips coupled `(1|p|species)` for nbinom2 in
our parity generator; R q=1 NB2 structured-σ covers scale-axis existence.
No `nbinom2-locscale` R fixture in this closeout.

`Tweedie random intercept (mean)` is `implemented` (#563 S8): an ordinary
`(1 | g)` random intercept, and (as of the same slice's second cell) an
independent `(0 + x | g)` random slope, on `mu` via 32-node Gauss–Hermite
quadrature (`src/tweedie.jl` `_fit_tweedie_ranef` / `_fit_tweedie_slope_ranef`,
the same scheme as the Poisson/Gamma `(1 | g)` routes), same-target checked
against drmTMB 0.7.0 (`test/test_tweedie_ranef.jl`, fixtures
`test/parity/fixtures/tweedie-mu-ranef/` and
`test/parity/fixtures/tweedie-mu-slope-ranef/`). Matches drmTMB's own scope
exactly: the CORRELATED random slope `(1 + x | g)`, random effects on
`sigma`/`nu`, and structured (phylo/relmat/animal/spatial) markers on `mu`
stay `rejected` in both packages (drmTMB's `validate_tweedie_mu_random_terms()`
/ `validate_tweedie_random_terms()`).

## Estimation and inference

| Capability | Status |
|---|---|
| REML (Gaussian fixed-effect location-scale) | implemented |
| REML with ordinary random effects (Gaussian mean) | implemented |
| REML bivariate phylogenetic location-scale (q4, all axes) | implemented |
| Conjugate-EM Gaussian phylo-mean (`algorithm = :em`) | implemented |
| Natural-gradient EM (`algorithm = :natgrad`) | rejected |
| Fisher / observed-info metric (`lc_metric`) | implemented |
| Wald SEs and CIs (observed information) | implemented |
| Profile-likelihood CIs | implemented |
| Parametric bootstrap CIs | implemented |
| AGHQ adaptive-quadrature marginal estimator | implemented |
| Variational (VA/ELBO) marginal estimator | planned |
| Chi-bar-square boundary LRT p-value | implemented |
| Model comparison suite (LRT/anova/AICc/weights/update) | implemented |
| Heritability/repeatability/ICC accessors | implemented |

`Natural-gradient EM (algorithm = :natgrad)` is `rejected` on measured evidence
(2026-08-01 #13 decision gate): `fit_em_natgrad` stalls at logLik ≈ −259.80 on
`q4_p100` vs `fit_q4_sparse_tmb` −256.51 (same class as plain block-coordinate
EM). Brief: `docs/dev-log/plans/2026-08-01-natgrad-decision-gate.md`. The
reusable Fisher metric was extracted as `lc_metric` (`src/lc_metric.jl`,
`test/test_lc_metric.jl`) — infrastructure for AI-REML / #11/#165, **not** a
public solver.

`REML with ordinary random effects (Gaussian mean)` is `implemented` (#439):
`src/gaussian_core.jl` admits a single Gaussian mean intercept `(1 | g)`
under opt-in `method = :REML` (the former 413–423 FE-only hole; σ-RE, slopes,
multi-ranef, and structured / phylo / meta stay `ArgumentError`), and
`src/gaussian_ranef.jl` adds the Patterson–Thompson term
`½ logdet(Xμ′ V⁻¹ Xμ)` on the Woodbury spine. Exercised by
`test/test_reml_ordinary_ranef.jl`, which is **in the default suite**
(`test/runtests.jl`, wired by #445 "Option A" after #439/#440 landed);
`test/test_reml.jl` retains the random-slope rejection. ML stays the
default. This is not AI-REML, not a TSV flip, and not “parity complete.”

`REML bivariate phylogenetic location-scale (q4, all axes)` is `implemented`:
`src/reml_q4.jl` is included in the module (`src/DRModels.jl:65`) and
`test/test_reml_q4_allaxes.jl` is in the default suite
(`test/runtests.jl`), asserting the restricted correction reaches all four
axes (mu1, mu2, sigma1, sigma2; issue #18 regression). The current public
capabilities page also lists REML for q=4 Laplace and Location–Scale–Scale
models as tested; the source-and-test citations here supply the implementation
trail behind that summary.

`Model comparison suite (LRT/anova/AICc/weights/update)` is `implemented`, but the
`weights` member needs reading carefully, because two things in this ledger point
different ways and a scanning reader will take the wrong one.

- It is **not** Akaike / model weights. Neither DRModels.jl nor drmTMB computes those,
  despite `weights` sitting in a list next to `AICc` — which is precisely the
  reading the row name invites. There is no `akaike_weights` in either package.
- It is `StatsAPI.weights`: **prior, per-observation** weights. `src/comparison.jl`
  returns `ones(nobs(fit))` unconditionally, because DRModels.jl fits do not store prior
  weights at all. Its docstring says so plainly.
- drmTMB's `weights.drmTMB` returns a real stored vector (`object$model$weights`,
  `R/methods.R`), which it genuinely uses — its bootstrap reads it
  (`R/profile.R`) and `associate_pairs()` guards on `any(weights != 1)`.

So the accessor is at parity in *name* and not in *substance*, and the gap is
already recorded elsewhere in this ledger: the `base_weights` gate is closed as an
`intentional_error` on the grounds that "DRModels.jl bridge payload has no weights
slot". Passing `weights = ...` through `engine = "julia"` is refused.

Kept `implemented` because every named member exists and is exported, which is this
file's stated bar. Flagged because a row can meet the bar and still leave a reader
believing something false — and the fix for that is prose, not a status change.

`Chi-bar-square boundary LRT p-value` is `implemented`: `src/chibar.jl` is
included (`src/DRModels.jl:161`), exports `chibar_pvalue`/`lrt_boundary`
(`src/DRModels.jl:160`), and `test/test_chibar.jl` is in the default suite
(`test/runtests.jl`). The current public capabilities page lists chi-bar-square
boundary inference as tested.

`AGHQ adaptive-quadrature marginal estimator` is `implemented` (2026-08-24
audit, PR #449 / commit `93c3db6b`, merged 2026-08-18): `src/aghq_1d.jl` is
included at `src/DRModels.jl:91` and wires a public front end on `drm()`
(`marginal = :AGHQ`, Poisson `(1 | g)` only — `src/poisson.jl:35-37,176-177`).
`test/test_aghq_1d.jl` is in the default suite (`test/runtests.jl`) and
exercises the quadrature kernel, the public fit path, and the fail-loud
guards on every unsupported structure (phylo, crossed, correlated slope,
other families, `:REML`, `associate_pairs`). This corrects the prior
`missing` cell: the same PR that landed the code also wrote the `missing`
row and its "not ADEMP-certified" rationale, but by this file's own ladder
(source + a registered test) that is source-and-test evidence for
`implemented`, not for `missing`. Scope stays exactly what #448 shipped —
Poisson `(1 | g)` only, `:REML` not wired to `:AGHQ`, tensor/multi-d AGHQ on
phylo Laplace out of scope — this flip changes the status word only, not the
scope.

`Variational (VA/ELBO) marginal estimator` stays `planned` here, but the
citation needs correcting: `test/test_variational.jl` is not plumbing-only
(anchors exercise real ELBO fits), and the public `marginal = :VA` front end
works — not just the generic `_fit_va` stub — for Poisson / Binomial /
NegBinomial2 / Gamma / Beta `(1 | g)` (`src/poisson.jl:26`,
`src/binomial.jl:38`, `src/negbinomial.jl:69`, `src/gamma.jl:34`,
`src/beta.jl:35`), each with its own registered test
(`test/test_va_poisson_elbo.jl`, `test/test_va_frontend_families.jl`,
`test/test_variational_binomial.jl`, `test/test_variational_nb2.jl`,
`test/test_variational_gamma.jl`; `test/runtests.jl`). Per
`docs/dev-log/check-log.d/2026-08-09-136-va-rung2-3.md`, the project's own
guide banner was corrected from "Planned" to "Experimental" for this reason,
and that entry explicitly notes "Does not close #136." This audit leaves the
chip word alone: issue #136 stays open (phylo, crossed, correlated slopes,
zi/hu, and 136e remain unwired, per `src/variational.jl`'s own docstrings),
and the owner's 2026-08-24 flip authorization named AGHQ specifically, not
VA. Treat this row with the same care as `:natgrad` and `#49` below: real
code exists for a real subset, but this audit does not upgrade the chip past
what the open issue supports without a separate, explicit call.

## Known sampling covariance (`meta_V`)

The exact-main contracts differ for the public scale accessor. At drmTMB main
`b535bfbcce2a9039d869c2e34fe9da476394bf62`, `sigma()` on a `meta_V(V=...)`
fit reports the fitted between-study heterogeneity SD `tau`; the supplied
sampling covariance remains separate in the likelihood and simulation. At
DRModels.jl main `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`, the univariate
`meta_V(v)` likelihood and `predict_parameters(... )[:sigma]` use `tau`, but
the fit stores only `sqrt(v + tau^2)` in `fit.scales[:sigma]`, which the
public `sigma(fit)` accessor returns. This is a current-main accessor parity
gap; the Julia meta-analysis tutorial describes `sigma` as `tau`.

The maintainer-selected contract for both packages is `sigma(fit) = τ` on
`meta_V` models. This agrees with drmTMB main and the dirty Julia candidate
below, but not with DRModels.jl exact main; candidate behavior is not yet a
merged parity claim.

The dirty candidate in this worktree retains `tau` separately as
`fit.scales[:meta_tau]`, reports it from `sigma(fit)` and bridge output, and
keeps `sqrt(v + tau^2)` internally for simulation. Its retained Oct. 28
receipt's seven Julia source/test hashes match this worktree; the recorded
focused tests, same-fixture point fit, and one live fixed-effect bridge fit
are candidate evidence. They do not change exact-main behavior or establish
interval calibration. See
`docs/dev-log/evidence/meta-v-sigma-20261028/receipt.md` and
`docs/dev-log/after-task/2026-10-28-meta-v-sigma-accessor.md`.

Rechecked 2026-11-01 18:03 UTC: all seven source/test SHA-256 values in that
receipt still match this dirty candidate, and every retained file listed in
its `SHA256SUMS` verifies. This renews the receipt-to-worktree identity check;
it does not rerun the tests. File preflight still finds four refs carrying
work on `src/gaussian_meta.jl`, so the candidate remains unmerged and source
ownership remains unresolved.

The separately reproduced Gaussian `meta_V` `simulate(newdata)` variance
omission is also repaired in the current dirty candidate, but not in exact
main. Candidate `src/gaussian_core.jl` and `test/test_meta.jl` hashes match
their source manifest and the focused receipt; Julia 1.10.12 on Totoro passed
13/13 focused assertions, including 12,000 new-data draws against
`sqrt(v_new + tau_new^2)` and rejection of missing or invalid sampling
variances. This qualifies the native fixed-effect simulation path on that
fixture, not R-to-Julia simulation transport, random-effect simulation, or
package-wide behavior. The candidate remains unmerged and ownership of the
overlapping source files is unresolved. See
`docs/dev-log/evidence/cross-repo-parity/meta-v-newdata-simulation-20261028.md`
and `docs/dev-log/after-task/2026-10-28-meta-v-newdata-simulation.md`. The
open documentation PR #1210 is a separate candidate; its reviewed head does
not contain this repair.

The [public Julia meta-analysis guide](https://itchyshin.github.io/DRModels.jl/dev/model-guides/meta-analysis.html)
currently states that `sigma(fit)` is `tau`, but the deployed site is based on
synthetic merge `645b435` and exact main still returns `sqrt(v + tau^2)`. This
makes the public sentence incorrect for the currently released main behavior;
the dirty candidate evidence above does not correct a published page. The
[README](https://github.com/itchyshin/DRModels.jl/blob/main/README.md) also
says version 0.7.1 although exact main's `Project.toml` is 0.7.2, and says the
package is not in General. The live registry lookup found no
`D/DRModels/Package.toml`, so the registry statement is consistent while the
version label is stale. The repository tag v0.7.2 and GitHub's latest release
v0.1.1 are separate records; do not infer a release from the tag. No external
release text or published page was changed.

The [drmTMB v0.7.0 GitHub release](https://github.com/itchyshin/drmTMB/releases/tag/v0.7.0)
still says it was not submitted to CRAN, while the [CRAN package record](https://cran.r-project.org/package=drmTMB)
lists v0.7.0 as published on 2026-09-15 and the current drmTMB README says
0.7.0 is on CRAN. This is release-description drift, separate from the
cross-language parity evidence; verify the tag and artifact identity before an
external wording correction.

Full univariate dense sampling covariance is a separate unresolved parity
cell. R main accepts a supplied dense `V`; Julia main's univariate `V = V`
route raises `ArgumentError` and directs users to the separate bivariate
`meta_vcov_bivariate` route or diagonal `meta_V(v)` route. The approved
compatibility design is an explicit covariance mode that preserves Julia's
current default and lets the R bridge select the R-compatible mode. That mode
is not present in the exact-main or inspected dirty source. No implementation
owner or open implementation PR was found in the reviewed coordination and
GitHub records; file-level source ownership remains unresolved, so this
capability is not promoted.

## Bivariate structure and missing data

| Capability | Status |
|---|---|
| Bivariate structured random effect on all four axes (q4 PLSM) | implemented |
| Cross-family bivariate (different families for y1 y2) | experimental |
| Bivariate Gaussian with ordinary correlated random effects, `(1 \| p \| g)` | rejected |
| Missing-response handling (native, per fitted route) | missing |
| Missing-predictor imputation (mi()) | experimental |
| R to Julia bridge (engine=julia) | implemented |

`Bivariate Gaussian with ordinary correlated random effects, (1 | p | g)` is
`rejected`: `drm(::BivariateDrmFormula, ::Gaussian)` throws an `ArgumentError`
that names drmTMB when `mu1`/`mu2`/`sigma1`/`sigma2` carry an ordinary `(1 | g)`
or `(1 | p | g)` term (`_split_bivariate_q4_rhs`, `src/gaussian_bivariate.jl`).
drmTMB fits this model, so it is a known R-side-only cell (#1094); only
`phylo`/`relmat`/`animal`/`spatial` markers are accepted on this route. The same
method also refuses a `marginal =` keyword other than the default `:LA` with an
`ArgumentError`. Both refusals are pinned by
`test/test_bivariate_refusals_1094.jl`.

`Bivariate structured random effect on all four axes (q4 PLSM)` is the
flagship verified engine (`src/sparse_phy.jl`, `src/takahashi_selinv.jl`,
`src/sparse_aug_plsm.jl`, `src/fit_q4_sparse_tmb.jl`), with `Sigma_a` stored
on the fit. Providers: phylo tree (`test/test_gaussian_bivariate_phylo.jl`) and
level-indexed `relmat` / `animal` / fixed-range `spatial` via
`make_problem_from_Q` (`test/test_gaussian_bivariate_q4_structured.jl`, #189).
`Cross-family bivariate` is `experimental`, not `missing`: `src/mixed_family.jl`
(shared-latent GHQ across two different families) and
`src/mixed_family_postfit.jl` are included by the module, and the formula front
end (`drm(f::BivariateDrmFormula, fams::Tuple; data, …)`) replaces the prior
hand-built-design-only route. The default suite includes
`test/test_mixed_family.jl`, `test/test_mixed_family_postfit.jl`, and
`test/test_cross_family_formula.jl`. The recorded A11 decision explicitly
kept this capability `experimental` after the formula API and tests landed;
see `docs/dev-log/check-log.d/2026-08-16-a11-cross-family-formula.md`. Keep
that boundary until it is revisited. This status does not claim current R
parity, interval calibration, or broader random-effect support. The related
"single `gaussian_bivariate.jl` bivariate source file" statement in
`docs/src/capabilities.md` is stale and should be reconciled separately.

`Missing-response handling (native, per fitted route)` stays `missing`, with
two citation corrections. First, `src/missing_data.jl` is included at
`src/DRModels.jl:131`, not `:101` (`:101` is `mixed_family.jl`, above). Second,
"explicit listwise (complete-case) deletion only" undersells what exists:
`_fit_observed_response_rows` (`src/gaussian_core.jl:740`) is a shared helper
used by twelve family files (`beta.jl`, `betabinomial.jl`, `binomial.jl`,
`cumulative.jl`, `gamma.jl`, `gaussian_core.jl`, `lognormal.jl`,
`negbinomial.jl`, `poisson.jl`, `tweedie.jl`, `student.jl`,
`zeroonebeta.jl`) that auto-drops missing/NaN-response rows inside `drm()`
itself, with a warning, for every one of those families -- no separate
`drm_listwise` call needed. Separately, `leaf_nll` in `src/sparse_aug_plsm.jl:37`
(the flagship q4 bivariate phylo engine, wired via `src/fit_q4_sparse_tmb.jl`
at `src/DRModels.jl:40`) takes per-cell `o1`/`o2` observed flags and evaluates the
correct univariate marginal when only one axis is observed -- a genuine
masked partial likelihood, not row deletion. Registered tests:
`test/test_missing_response.jl`, `test/test_missing_response_nongaussian.jl`
(fourteen families), and `test/test_missing_response_bivariate.jl` (FD-vs-exact
gradient with masked cells, plus a missing-at-random fit check)
(`test/runtests.jl`). This is still short of drmTMB's named row --
a native masked likelihood across 18 fitted routes -- because outside the q4
bivariate engine the native mechanism is auto-triggered listwise deletion,
the same underlying operation as `drm_listwise`, not a masked likelihood;
issue #49 remains open and its own file header states FIML for missing
responses is explicitly out of scope. This audit leaves the chip as `missing`
given #49 is parked and the owner's 2026-08-24 authorization did not name
this row, but corrects the stale citations and the "listwise deletion only"
undercount above. `Missing-predictor imputation (mi())` was `missing` before
2026-08-30 (#563); that is now stale. Real code exists AND is exported AND is
tested: `mi`, `JointDrmFit`, `JointTwoDrmFit`, `JointFiniteDrmFit`, `imputed`,
`miss_control`, `impute_model` are exported from `src/DRModels.jl:193-197`, backed
by six files at `src/DRModels.jl:137-143` (#563), and covered by
`test/test_joint_missing_*.jl` (auto-discovered by `test/runtests.jl`). It is marked
`experimental`, not `implemented`, because D-181 (2026-08-28, reaffirmed by
D-209 §3, 2026-09-02) explicitly fences the mi() axis out of the v1.0 twin
claim, and because the scope itself is narrow: a Gaussian response with one
Gaussian-or-Bernoulli `mi()` predictor, two independent Gaussian `mi()`
predictors, or one ordinal/categorical `mi()` predictor, each with complete
remaining fixed-effect designs, no random/structured effects, no REML, and no
profile/bootstrap intervals (`docs/src/reference/model-specification.md`,
`docs/src/reference/engine-internals.md`). No R-parity claim is made for this
route.

`R to Julia bridge (engine=julia)` is `implemented` as a public bridge surface:
`src/bridge.jl` exports `drm_bridge`/`drm_bridge_inference` (`src/DRModels.jl`
export list), and `test/test_bridge.jl` checks bridge output against native
`drm` output. This status does not claim parity for every route. The current
drmTMB `inst/extdata/julia-capabilities.tsv` at `main` commit
`b535bfbcce2a9039d869c2e34fe9da476394bf62` has 37 routes: 17 supported,
6 partial, 13 experimental, and 1 unsupported on its `r_bridge_status` axis.
Its separate `claim_status` axis has 10 covered, 22 partial, 4 experimental,
and 1 unsupported. These counts are a snapshot of that exact source, not a
permanent capability total; the labels are not interval-coverage results.
Consult the exact-source route record before making a parity or uncertainty
claim.

The hurdle-NB2 response-scale output remains a concrete partial bridge route.
R main's route row `hurdle_nbinom2` is `partial` on both bridge and claim
axes. On exact main, the Julia bridge's `_bridge_fitted_marginal` at
`src/bridge.jl:2698-2708` applies the zero-inflation correction only and leaves
`hu` uncorrected. R's `drm_fitted_response()` at `R/methods.R:6695-6699`
computes the hurdle-adjusted mean. A matched Totoro diagnostic using current
R `b535bfbcce2a9039d869c2e34fe9da476394bf62` and Julia
`64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716` measured a maximum dpar difference
of `9.77e-13` and log-likelihood difference of about `2.6e-11`, while the
maximum fitted-value and response-residual differences were both `1.2416`.
Both fits converged. The R response mean reconstructed from Julia's three
dpars agreed with native R to `7.67e-13`; the Julia bridge fitted value agreed
with its `mu` dpar to `4.44e-16`. The exact-source receipt, environment,
runners, and logs are in
`docs/dev-log/evidence/cross-repo-parity/hurdle-nb2-20261101/receipt.md`.

Both current-main test trees make this boundary explicit. Julia's
`test/test_bridge_zi_marginal_mean.jl` hurdle case asserts that bridge
`fitted()` equals the `mu` dpar and residuals equal `y - mu`, preserving the
component-mean behavior. R's
`tests/testthat/test-julia-family-hurdle_nbinom2.R` tests the R-shaped bridge
result against `hurdle_nbinom2_mean(mu, sigma, hu)` and currently asserts that
Julia bridge `fitted()` is the raw `mu` value and differs from that R target.
The native R fixture separately checks fitted and residual formulas against
an independently expanded zero-truncated NB2 mean and variance. The exact-main
paired bridge test file itself currently errors before its fit-output
assertions because `fixef()` has no applicable method for `drmTMB_julia` in
this load order. A diagnostic runner that avoids that extractor successfully
measured the model outputs, but this does not turn the standard test file
green. The route stays partial pending an owner-reviewed bridge correction,
the separately tracked `fixef()` registration fix, and a passing exact-source
test that checks all three dpars, fitted values, residuals, and a zero-inflated
NB2 control.

At drmTMB main (reviewed 2026-11-01),
`b535bfbcce2a9039d869c2e34fe9da476394bf62`, `R/zzz.R` registers `fixef` and
`ranef` only for the native `drmTMB` class in `nlme`'s S3 table. The two Julia
bridge classes are not registered there, so `fixef()` can fail for a bridge
fit when `nlme`'s exported generic masks drmTMB's re-export. Open
[#1435](https://github.com/itchyshin/drmTMB/pull/1435) proposes registering
both bridge classes; its head is `d012cc28593691168167e07c3f8e0ff9122ccc61`.
It remains open and non-draft. The PR body reports six checks, but a direct
workflow lookup returned no run records, so those checks remain owner-reported.
The fix is not in pinned main. Treat it as a candidate until the source lands
and loaded-package dispatch is verified.

## Current-source validation debt for broad parity claims

## Open cross-language parity PR evidence (reviewed 2026-11-01)

Current main remains DRModels.jl `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`
and drmTMB `b535bfbcce2a9039d869c2e34fe9da476394bf62`. The following open
PRs add candidate parity evidence; none changes the main-source status above
until it is merged and the resulting source is checked. GitHub showed no
assignees, review decisions, or review comments on these PRs at this review.
The later lane-comment refresh below is separate from submitted reviews and
inline review threads.

| Candidate | Exact head and observed checks | Review of the evidence and its limit |
|---|---|---|
| DRModels.jl [#1242](https://github.com/itchyshin/DRModels.jl/pull/1242), Poisson/NB2 fixed effects and `(1 | g)` | `c56ecfb3f528287bde3e179846456ba9a8fff7ca`; full Julia shards and `ci-ok` passed in run `41249837752`; docs preview passed in `41249837766`. | Test-only. PR body reports fixed-effect and explicit `:Laplace` agreement, while nine default-integrator comparisons remain `@test_broken` because DRModels uses adaptive quadrature and drmTMB uses Laplace. The body still says CI has not run; live checks supersede that sentence. Do not claim default-route parity. |
| DRModels.jl [#1239](https://github.com/itchyshin/DRModels.jl/pull/1239), Beta/Binomial/BetaBinomial | `e8a6c20a8c19da2aa1616b4f33dc8df7b73ea34e`; full Julia shards and `ci-ok` passed in run `41248334383`; docs preview passed in `41248334332`. | Test-only. PR body reports three `@test_broken` marginal log-likelihood comparisons under different integrators; coefficients and random-effect SDs are closer. It does not change the integration method or establish parity for the default marginal likelihood. |
| DRModels.jl [#1234](https://github.com/itchyshin/DRModels.jl/pull/1234), Student/Gamma location-scale | `0b85b7cf2fb47b2a12d8edb555494ff2705614ee`; full Julia shards and `ci-ok` passed in run `41247508362`; docs preview passed in `41247508281`. | Candidate test evidence only. The Student fixture retains four expected mismatches at the Gaussian-limit boundary; the Gamma fixture is still skipped because its reference output has not been generated. Green CI does not mean these parity cells passed. |
| DRModels.jl [#1240](https://github.com/itchyshin/DRModels.jl/pull/1240), Gaussian location-scale with `(1 | g)` | `03b93e680266369dff10e63eb4c846826a102ec4`; run `41248908491` failed on Julia shard 2 and `ci-ok`. | Candidate not admitted. The failing test is `test_fixture_provenance.jl`: the new `expected.meta.toml` lacks required `drmtmb_code_hash`. This blocks provenance validation before the Gaussian location-scale assertions; it is not evidence of a model mismatch. PR remains draft and blocked. |
| drmTMB [#1659](https://github.com/itchyshin/drmTMB/pull/1659), seven-family stored-reference tests | `1539b2b52df017aec9159e4d6ce288626e8f0b8b`; OS-matrix, source-tree tests, and all four release shards passed in run `41248855941`. | Strongest current candidate crosswalk addition: 14 fixed-effect/random-intercept cells use stored DRModels.jl values from the exact current Julia main pin above. The PR body reports a local live Julia bridge check for admitted cells, while the full local check without `DRMODELS_JL_PATH` skipped that live path. Six default-integrator comparisons remain skipped by design. One observed dataset per cell assesses numerical agreement, not recovery or interval coverage. |

These checks reconcile the PR heads and workflow states, not every lane's
scientific acceptance. The PRs are maintainer-authored, but GitHub has no
independent reviewer or owner-resolution record for the rows above. Retain
their exact candidate SHAs and limits when deciding whether to merge or
promote any route.

These are open statistical-validation boundaries, not assertions that the
exact-main engines currently fail in the same way:

- **Julia Gaussian q=4 phylogenetic covariance and scale-axis intervals:** the
  2026-08-25 campaign at Julia source `6f52d179` (an ancestor of current main)
  found q4 covariance off-diagonal conditional coverage `0.876`, including
  scale-by-scale entry `L43 = 0.827`, and scale-axis diagonal coverage about
  `0.997`. The campaign is historical: current main changed
  `src/gaussian_bivariate.jl`, `src/inference.jl`, and `src/vcov_guard.jl`
  substantially since that pin. No exact-current-main calibration result was
  verified here; do not reuse the historical values as current performance.
- **R NB2 phylogenetic q1 mean recovery and intervals:** exact drmTMB main's
  `docs/design/34-validation-debt-register.md` and
  `docs/design/46-pre-simulation-readiness-matrix.md` keep
  `nbinom2_phylo_q1_mu` partial/held because profile intervals fail often at
  the true-zero phylogenetic-SD boundary and low-count fixed-scale recovery is
  unstable. Its smoke fits and other provider routes do not close that gate.
- **R bivariate covariance and spatial mesh:** the same exact-main debt
  register keeps q4/q6 formal recovery limited and q8 coverage/power blocked;
  its 2026-06-07 q8 diagnostic has zero positive-Hessian rate and no usable
  Wald intervals. The `spatial_mu_coord` readiness row also keeps the n=64
  mesh recovery rung outside point-fit recovery after it failed. Reconcile the
  original source and manifests before using these historical artifacts as
  positive evidence.

The next statistical step is a source-pinned paired q4 Gaussian phylogenetic
bivariate REML admission. First freeze the estimand and parameter map, validate
the corrected matrix-normal simulator, and compare a tiny fixed-parameter
objective and Hessian with an independent dense oracle. Then estimate runtime
on a short exact-source pre-run before deciding whether a 1,000-replicate
current-pair replay fits under the three-hour gate. That replay must retain all
attempts, failed or unavailable intervals, positive-information status,
coverage, width, and Monte Carlo error. Curie's read-only audit recommends this
route because the historical q4 inference source has changed substantially;
it is a proposed next gate, not work run or approved in this reconciliation.

## Snapshot

- 51 capabilities, all `implemented`/`rejected`/`planned`/`missing`/
  `experimental` per the mapping above; 45 `implemented`, 2 `rejected`
  (`:natgrad` and ordinary correlated bivariate random effects), 1 `planned`,
  2 `experimental` (`Missing-predictor imputation (mi())` and cross-family
  bivariate), and 1 `missing` (native missing-response handling).
  This recount adds the ordinary non-Gaussian Laplace and coupled NB2
  location-scale rows above. It replaces the stale 46-capability snapshot
  while retaining the dated audit history below. (2026-08-24 chip audit flips `AGHQ
  adaptive-quadrature marginal estimator` `missing` -> `implemented`: PR #449
  / commit `93c3db6b` landed source wired into `src/DRModels.jl` plus a test
  registered in `test/runtests.jl`, meeting this file's own ladder. The same
  audit re-examined `Cross-family bivariate`, `Missing-response handling
  (native, per fitted route)`, and `Variational (VA/ELBO) marginal
  estimator`, found source+test evidence undercounted in each, corrected the
  stale citations, and left all three chips unflipped for documented reasons
  -- see `docs/dev-log/evidence/2026-08-24-chip-audit.md`. Prior snapshot
  said 37/1 while the table still listed two `rejected` rows; that recount
  flipped the ordinary-RE REML chip and left `:natgrad` as the only
  `rejected` row. 2026-09-02 (S9, #563): `Missing-predictor imputation
  (mi())` flips `missing` -> `experimental` -- the joint missing-predictor
  routes exported at `src/DRModels.jl:190-197` are now real code, exported, and
  tested, but D-181/D-209 §3 fence them out of the v1.0 twin claim, so they
  are labelled `experimental` rather than `implemented`.)
- Export-name audit of drmTMB `main` `b535bfbcce2a9039d869c2e34fe9da476394bf62`
  against DRModels.jl `main` `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`
  (2026-11-01): 73 R exports and 197 Julia exports; zero unaccounted R export
  gaps after documenting 15 raw name mismatches; 40 additional Julia exports
  have no R twin. This is name-level accounting, not evidence of matching
  behavior. The 37-row R route ledger above remains the route-level status
  source, with separate bridge-support and claim-evidence axes.
- Main-CI refresh (2026-11-01 15:54 UTC): Julia run
  [41315894403](https://github.com/itchyshin/DRModels.jl/actions/runs/41315894403)
  is on `64ccb2d8d87b329e5a43b2358fd5cb4a36cd9716`. Julia 1.10 shard 2
  failed at `test/test_bootstrap.jl:105-106` (one usable refit and one failed,
  rather than two usable). Its exact job log records 44 passes and these two
  assertions failing in the phylogenetic bootstrap block. Julia 1.10 shard 1
  and Julia 1 shard 1 were still running at this refresh; the other six
  completed Julia shards passed. A 15:54 UTC recheck found those same two
  shards still in progress, with Julia 1.10 shard 2 still failed. A 16:01 UTC
  read found the same run still in progress, with those two shards active and
  the same shard-2 failure. A 16:16 UTC read found no status change; the run
  remained in progress with the same two shards active and one failed shard.
  No rerun was requested. At 17:01 UTC, run `41315894403` concluded failure;
  six of its eight Julia shards passed. In addition to
  the bootstrap fixture failure in shard 2, Julia 1.10 shard 1 failed the
  end-to-end clamp assertion at `test/test_j3a_gradient_hessian_vcov.jl:158`:
  the non-clamped diagonal entries of `vcov(fit)` were not all finite. This is
  the seeded NB2 crossed fit with one random-effect SD at its clamp. The test
  file blob matches this exact main checkout; no candidate-only change
  explains the failure. A scratch-clone reproduction on Julia 1.10.12 now
  narrows the cause: the fit is marked nonconverged, its largest absolute
  free-coordinate gradient is `5.15e-3`, and its two random-effect log-SDs
  are `-6.389` and `-8.059` (the latter at the lower clamp). The covariance
  path excludes the clamped coordinate, but the cold-started gradient Hessian
  of the remaining four coordinates has a smallest eigenvalue of `-1.36e-5`;
  the value-based fallback also reports a non-positive-definite Hessian. The
  ordinary stored-gradient Hessian is positive definite, so this is a
  near-boundary disagreement in the SE calculation, not evidence that the fit
  has a trustworthy covariance. The result is consistent with withholding
  Wald SEs for this nonconverged fit near a second collapsing variance. The
  current assertion expects finite SEs without checking convergence or Hessian
  status; an owner should review that expectation before any test or engine
  change. The full Julia CI run is therefore failed, not pending, and
  this result is separate from the two green candidate PR checks described
  below. No interval-calibration claim follows from either CI result.
- Issue and owner reconciliation (2026-11-01): Julia issue
  [#1190](https://github.com/itchyshin/DRModels.jl/issues/1190) remains open
  with no assignee; its R twin
  [#1594](https://github.com/itchyshin/drmTMB/issues/1594) is closed as
  completed. The current Ubuntu exact-main failure in run `41315894403`, job
  [124875847159](https://github.com/itchyshin/DRModels.jl/actions/runs/41315894403/job/124875847159),
  fails only the assertion that every non-clamped `vcov` diagonal be finite
  for a fit already marked nonconverged. The retained Julia 1.10.12 scratch
  reproduction has log SDs `-6.389` and `-8.059`, gradient maximum `5.15e-3`,
  and a non-positive Hessian on both the gradient and value paths. This does
  not establish the #1190 acceptance failure, which concerns a fit reported
  converged with unusable SEs. It also conflicts with an earlier #1190 comment
  that the Linux run of this test had a positive-definite Hessian. The
  crosswalk leaves the assertion and engine unchanged pending owner review.
  No assignee or open source-fix PR for this J3a case was found. File preflight
  still finds two refs: the Oct. 16 J3d commit concerns a separate non-PD
  phylo-mean Hessian fallback (Refs #1073) and does not supply the current
  clamp-case fix; `j3a-red-test-only` is explicitly throwaway. Neither resolves
  the current failure. No current owner resolution is recorded.
  Open PR
  [#1216](https://github.com/itchyshin/DRModels.jl/pull/1216), head
  `2d4dc87294d5c6e67b1f834c767c24639ddd0987`, remains open and mergeable over
  this main. Exact-head CI run
  [41152894330](https://github.com/itchyshin/DRModels.jl/actions/runs/41152894330)
  passes all Julia shards and `ci-ok`; Documenter run
  [41152894311](https://github.com/itchyshin/DRModels.jl/actions/runs/41152894311)
  passes its docs build and preview (deploy is skipped). The PR changes the
  phylogenetic bootstrap test fixture from two to
  four observations per tip; its body records the seed-13 refit reaching
  `log(sd_phy) ≈ -3.18`, being dropped as nonconverged, and states that the
  fixture change does not fix the EM fit. It is unmerged and does not establish
  bootstrap interval calibration.
- Local focused reproduction from an archive of that exact main SHA, on Julia
  1.10.12 with one Julia and one BLAS thread, reproduced the `m = 2` test
  failure: attempted 2, used 1, failed 1. The recorded failed replicate had
  seed `0xde00a4b8b6afb00d` and message `refit did not converge or landed on a
  degenerate optimum`; the test had 1 pass and 2 failures. On the same source,
  dependency environment, and seed inputs, changing only the fixture to
  `m = 4` passed 3/3 assertions with no failed replicate. This confirms the
  fixture change avoids this one observed case. It does not distinguish a
  solver stopping-rule defect from a genuine boundary optimum, and does not
  establish broader optimizer reliability or interval calibration. The
  temporary environment and runners are described in the dated after-task
  report.
- Open draft [#1226](https://github.com/itchyshin/DRModels.jl/pull/1226), head
  `26087694a72ea645aa3191752dbd32f0fc0ae758`, targets the same main SHA and
  makes the same `m = 2` to `m = 4` fixture change in
  `test/test_bootstrap.jl`. Its exact-head CI run
  [41244566371](https://github.com/itchyshin/DRModels.jl/actions/runs/41244566371)
  passed all four Julia shards; Documenter run `41244566368` also passed. Its
  body reports 14/1,000 boundary refits at `m = 2` and 0/1,000 at `m = 4` on
  Julia 1.10, with no solver-code change. PR #1216 also changes this fixture
  while carrying a separate bootstrap behavior patch. These two open PRs
  overlap on the same test file; no owner resolution or merge is recorded here.
  Issue #1224's latest comment asks to keep the issue open to assess whether
  the `m = 2` EM stop is a boundary optimum or a stopping-rule defect, and says
  the fixture change only masks that case in this test. The current #1216 body
  still has an older sentence saying no open issue tracks the boundary, but
  appended text now references #1224; the issue link is present while that
  sentence remains internally stale. The maintainers have not recorded a
  resolution of their duplicate fixture edit; keep #1226's fixture-only patch
  separate from #1216's bootstrap-status change until they coordinate. The
  `m = 4` fixture does not prove interval calibration.
- Live PR refresh (2026-11-01): #1216 remains open, non-draft, and mergeable
  at `2d4dc87294d5c6e67b1f834c767c24639ddd0987`; its owner comment says to
  ship after exact-head CI passes and the body correctly references #1052.
  Workflow API records independently confirm its CI and Documenter runs are
  successful at that SHA. #1226 remains open, draft, and mergeable at
  `26087694a72ea645aa3191752dbd32f0fc0ae758`; its CI and Documenter runs are
  also successful at that SHA. It still overlaps #1216 on
  `test/test_bootstrap.jl`, has no owner comment, and no resolution of the
  duplicate edit is recorded. No owner message was sent.
- drmTMB main / candidate check refresh (2026-11-01): main is
  `b535bfbcce2a9039d869c2e34fe9da476394bf62`. Its latest workflow run is
  scheduled pkgdown run
  [41217879763](https://github.com/itchyshin/drmTMB/actions/runs/41217879763),
  successful on that SHA. Receipt-staleness run
  [40681027801](https://github.com/itchyshin/drmTMB/actions/runs/40681027801)
  failed on the same main SHA at the receipt detector. The latest full
  `R CMD check` run found was
  [40375071745](https://github.com/itchyshin/drmTMB/actions/runs/40375071745),
  failed on older SHA `02095c67d7e70bbb648f45d2bbdd5aa0b1ea7be5`. No full
  current-main CI `R CMD check` result was found. A local full check of an
  archive of exact main, using `NOT_CRAN=true`, R 4.6.0, and macOS arm64,
  reached `testthat` and reported one failure among 29,548 expectations: the Bernoulli-by-Bernoulli
  pair-association symmetry assertion in
  `tests/testthat/test-associate-pairs-bernoulli-bernoulli.R:40` compares
  forward and reversed `eta` values `0.551583976` and `0.551583956` at
  tolerance `1e-8`. Its test-file SHA-256
  `f487548071de2808d3f546bbeb90a83419beb7e66d6ff1b4b86846c484ec5be4`
  matches the file on exact `origin/main`. The check reports 29,547 passes,
  one failure, 102 warnings, and 473 skips; 72 live Julia bridge tests were
  skipped by this local configuration. It finished with `Status: 1 ERROR`;
  vignette rebuilding passed. The built tarball and raw logs are retained at
  `/Users/z3437171/Documents/Codex/2026-11-01/drmtmb-main-check-b535bfbcce2a/`;
  the tarball SHA-256 is
  `2b1ef044cb452e967fad5faca202f736058a1415de4d02189fc09841b4a9c50a`.
  The failure is in the R main source, not a Julia bridge comparison, and no R
  source or test change was made here. Open #1435 and #1653 remain
  candidates over current main. Their latest PR comments report six passing
  checks on their respective heads (`d012cc28593691168167e07c3f8e0ff9122ccc61`
  and `181ae21ed00f5a85682509b6c7397eda15f3339a`), but direct workflow lookup
  returns no run records for those heads; treat the checks as owner-reported.
  #1435 is open, non-draft, and mergeable; #1653 is open, draft, and mergeable.
  Neither candidate changes the current-main crosswalk until merged and
  verified. Open draft #1652 (`deb5246002d3f2c6a97a1c5a111e83f378b5c232`)
  changes only the Bernoulli association symmetry test tolerance from `1e-8`
  to `1e-7`; the owner reports six green checks, but direct workflow lookup
  also returns no records. It directly addresses the local main test failure
  without changing model behavior; the tolerance still needs maintainer review.
- drmTMB PR #1627 remains open and draft at
  `36dc7a70cabcf0dec8428af4bd14fa2e0fc0625b` over current main. Its latest
  exact-head run
  [41272767129](https://github.com/itchyshin/drmTMB/actions/runs/41272767129)
  failed all four Ubuntu release shards at the generated-ledger validation
  step; source-tree tests and OS-matrix passed. The identical failure says
  `mc-0568`'s receipt blob for `R/drmTMB.R` (`8b715877f3be24c9fa4aaffb53f4b1c431780f70`)
  differs from the current blob (`b887d0d00573544f12607671288aa2dbe8423a93`),
  while the authenticated model-15 source fingerprint still matches. The
  validator calls this a stale receipt and directs its owner to rerun the
  committed source-equivalence runner. Comparing Git blobs confirms the
  receipt hash is the `R/drmTMB.R` blob on both current main
  (`b535bfbcce2a9039d869c2e34fe9da476394bf62`) and the PR base; the candidate
  alone changes it to the reported current hash. Thus the guard failure is
  candidate-specific file provenance drift, while the checked model-15 surface
  fingerprint is unchanged. The four jobs stopped before `R CMD check`, so this
  run does not establish a package-test failure. The PR has no assignee or
  owner comment in the reviewed metadata; its lane still needs to regenerate
  and review the receipt before the full checks can run.
- Live PR refresh (2026-11-01): #1627 is still open and draft at the same
  exact head. Its failed run remains the only retrievable exact-head workflow
  record. #1435 and #1653 remain open at the hashes above; their latest owner
  comments call them ready, with #1653 explicitly still draft. #1652 is also
  still draft. No R lane ownership or merge resolution was inferred from
  these comments, and no message was sent.
- Exact-head PR and lane-note refresh (2026-11-01 18:03 UTC): current GitHub
  metadata still shows Julia #1210 (open, non-draft), #1216 (open, non-draft),
  #1226 (open, draft), #1234 (open, non-draft), #1239 (open, non-draft),
  #1240 (open, draft), and #1242 (open, non-draft), all at the exact heads
  already recorded above. Exact-head Actions records are green for #1210
  (CI and Documenter), #1216 (CI and Documenter), #1226 (CI and Documenter),
  #1234 (CI and Documenter), #1239 (CI and Documenter), and #1242 (CI and
  Documenter). #1240's CI run `41248908491` failed while its Documenter run
  `41248908497` passed; the existing provenance failure description remains
  applicable. R #1659 remains open, non-draft, at `1539b2b52df017aec9159e4d6ce288626e8f0b8b`,
  with exact-head `R-CMD-check` run `41248855941` successful. These checks are
  candidate checks and do not change current-main parity status.
- The #1216 lane's latest comment (`cursor[bot]`, 2026-10-31) says it is ready
  for maintainer review and records green exact-head CI and Documenter runs.
  #1226 remains draft with no PR conversation comment; it still overlaps
  #1216's `test/test_bootstrap.jl` fixture change. No owner resolution of that
  overlap is recorded. For R, #1435 is open and non-draft, #1652 and #1653 are
  open and draft, and #1627 is open and draft. The latest Cursor lane comments
  on #1435, #1652, and #1653 report six green checks each; commit-run queries
  returned no records for those three heads, commit-status queries were empty,
  and direct Actions run URLs returned HTTP 404. Treat those checks as
  lane-reported, not independently verified. #1435's body still says “Draft
  only” despite its non-draft metadata; its latest lane comment says it is
  ready for Shinichi and “Do not merge from this agent.” No merge or owner
  resolution is inferred. #1627's exact-head run `41272767129` remains failed
  at the generated-ledger provenance gate, before `R CMD check`; its PR
  conversation has no owner status comment. #1659's successful run is
  independently retrievable. No external message was sent.
- Sources read: `src/DRModels.jl` (include list + export list), `README.md`,
  `docs/src/capabilities.md`, `docs/src/families.md`, `test/runtests.jl`
  (default-suite include list), and targeted `grep`/`git log` against
  `src/gaussian_core.jl`, `src/gaussian_ranef.jl`,
  `test/test_reml_ordinary_ranef.jl`, `src/reml_q4.jl`, `src/chibar.jl`,
  `src/missing_data.jl`, `src/skewnormal.jl`, `src/variational.jl`,
  `src/aghq_1d.jl`, `src/mixed_family.jl`, and `src/sparse_aug_plsm.jl` to
  verify claims the docs page did not make or got stale on.
- The current `docs/src/capabilities.md` lists SkewNormal fixed effects,
  q=4 REML, and chi-bar-square boundary inference, matching the source-backed
  statuses above. It summarizes tested support; it does not replace the
  route-specific limits in this ledger.
