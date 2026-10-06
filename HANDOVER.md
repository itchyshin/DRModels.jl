# DRModels.jl — Project Handover & Roadmap

> **Reader:** whoever continues `DRModels.jl` (human, Codex, or Claude). **Purpose:**
> hand over a *verified* proof-of-concept engine plus the full journey, the
> architecture, the traps, and the exact ordered path to v0.1.0 — so you start
> from knowledge, not from scratch. Every number here was reproduced by an
> independent run; don't lower that bar.

---

## 0. TL;DR

- **What:** `DRModels.jl` is the Julia twin of the R package **drmTMB**
  (univariate & bivariate *distributional* regression — a formula per parameter
  μ/σ/ρ). It mirrors the **gllvmTMB → GLLVModels.jl** move. Sister to GLLVModels.jl.
- **Why it exists:** drmTMB's selling-point model — the **q=4 phylogenetic
  bivariate location–scale model (PLSM)** — is wall-clock-bound by R/TMB in
  simulation studies. A fast Julia engine makes bootstrap / coverage / power
  studies feasible.
- **Verified:** **2.18× faster than drmTMB** on the q=4 single fit (same data,
  same marginal), **near-perfect O(p) to 10,000 species**, and **valid confidence
  intervals where drmTMB's Hessian fails entirely**.
- **State:** published at **https://github.com/itchyshin/DRModels.jl** (MIT, public);
  **v0.1.0 and v0.1.1 tagged**. The `drm()` / `bf()` front end is wired, all
  **13 families** (12 univariate + bivariate Gaussian) are implemented, exported,
  and recovery-tested, and **inference is wired** in `src/inference.jl` (Wald +
  profile + bootstrap; `infer_q4` graduated). Since the original handover, several
  comparison-suite capabilities have **merged and are now wired/exported**:
  **opt-in REML** (`method = :REML`, Gaussian location–scale, #11/#235) with a
  model-selection guard against the REML trap; the **conjugate-EM** Gaussian
  phylo-mean solver (`algorithm = :em`, promoted from `experimental/location_only.jl`,
  #12/#224); **heritability / repeatability / ICC** accessors with CIs (#233);
  **DHARMa-style randomised quantile residuals** for all families (#183/#234); and the
  **coevolution q=4 phylo front end** (#201) and its accessors (#187/#188).
  Julia-side R↔Julia helpers (`drm_bridge`, `drm_bridge_inference`, sparse-phylo
  companion) are in-tree; **Phase 1.5 / #5 is closed** at the experimental
  Hopper finish-matrix bar (#349 + drmTMB #878) — do not oversell as CRAN /
  “supported”. `src/experimental/` still holds **alternative estimator
  prototypes** outside the public API (SQUAREM / natural-gradient EM,
  trust-region & line-search E-steps, dense q=4 EM, warm-start fit variants,
  leftover `location_only.jl` copy). Public REML is `src/reml_q4.jl`
  (`method = :REML`) — there is **no** `experimental/reml_q4.jl` on tip.
- **Next:** Phase 1.5 / #5 **landed** (#349 + drmTMB #878; #5 CLOSED). Tip
  hygiene: keep claims off Julia General (**D-111** — catch up with drmTMB +
  both working well; drmTMB likely R/CRAN first). MIT via GitHub /
  `Pkg.develop`; do **not** pursue JuliaRegistrator. Longer horizon: optional
  deeper opt-in R-parity (`DRM_PARITY_TESTS=1`; #17 closed), Phase 3 articles,
  VA/ELBO (#136 open — Experimental `(1 | g)` on Poisson/Binomial/NB2/Gamma/Beta;
  136e / phylo/ZI still deferred). **Version / tag:** tree + tag **`0.1.2` / `v0.1.2`**
  (#346) — **not** General membership.

---

## 1. The mission

Build **the fastest mixed-model fitting engine** for the drmTMB model class, in
Julia, so the simulation studies that are bottlenecked by the R/TMB engine
(ADEMP coverage grids, parametric bootstrap, power) become cheap. The precedent:
the team's GLLVModels.jl pilot hit ~340× over R/gllvmTMB on the closed-form Gaussian
case. The open question this PoC answered: **can Julia also win on the *hard*
model** — the q=4 PLSM (Nakagawa et al. 2025 MEE, Model 5), where a shared
phylogenetic random effect drives `(μ1, μ2, log σ1, log σ2)` and the nonlinear
scale dependence means **no closed-form marginal** (it needs a Laplace
approximation)? brms/Stan: ~122 h. drmTMB (R/TMB): ~2.5 s at p=100. **Answer:
yes — 2.18× at p=100, and it scales where the alternatives can't.**

---

## 2. Bottom line — verified results

Same model, same real `q4_p100` data, same Laplace ML marginal as drmTMB.
Reproduced independently (`bench/run_sparse_tmb_nd.jl`, `bench/run_scaling.jl`;
inference now in `src/inference.jl`). Full grid + caveats: `report/comparison-grid.md`.

| Claim | Number | Status |
|---|---|---|
| **Single fit p=100 vs drmTMB** | **1.14 s vs 2.48 s = 2.18×**; logLik −256.51 ≈ −256.52; converged (drmTMB: "false convergence 8") | ✅ head-to-head |
| **O(p) scaling** (per-dim-variance, nrep=4) | p=100/1k/5k/**10k** = 0.77 / 4.5 / 49.5 / **112.9 s**; **k≈1.08** | ✅ verified |
| **Location-only (conjugate)** | **EM 3.1× faster than LBFGS** (p=200/1k), same MLE | ✅ conjugacy thesis |
| **Inference at the variance boundary** | Wald valid **16/17** params (drmTMB sdreport = all-NaN); bootstrap **60/60** | ✅ a genuine inferential edge |
| **REML mean-axis bias correction** | Λ[1,1] 0.70 → 0.83 (×1.18) | ⚠ mean-axis only |

**Defensible story:** DRModels.jl matches drmTMB's fit, runs 2.18× faster, scales
near-perfect O(p) (where gllvmTMB's multi-trait gradient is O(p²) — *their* code,
`sparse_phy_grad.jl`, "slope≈2"), EM wins on conjugate sub-models, and it returns
usable uncertainty where drmTMB's Hessian is singular.

**Do NOT oversell:** the historical “~12× vs drmTMB at p=10,000” figure was an
**extrapolation** and is **retired**. The measured nrep=4 head-to-head on Totoro
(#376; `docs/dev-log/evidence/2026-08-03-376-q4-scaling-h2h.md`) shows Julia
faster at p=100 (~3.3×) and drmTMB 0.6.0 comparable or faster at p≥1000 on that
synthetic ultrametric balanced grid — distinct from the real-data q4_p100
**2.18×** cell. The scaling curve uses the synthetic per-dimension-variance model
with replicates. The threaded-bootstrap speedup is **unmeasured** (the verified
run was serial, 39 s); warm-start is **not** a lever (tested: 0.8×).

---

## 3. The model

q=4 PLSM, the bivariate location–scale instance: responses `(y1, y2)`, with a
shared phylogenetic random effect on four axes `(μ1, μ2, log σ1, log σ2)` via a
4×4 between-species covariance Λ (a **separate phylo variance per axis** + cross-
covariances — biologically richer than one global variance), and a residual
correlation ρ. The public drmTMB-shaped front end is
`bf(mu1=…, mu2=…, sigma1=…, sigma2=…, rho12=…)`, with matching
`phylo(1 | species)` markers required on the four location/scale axes for the
q=4 route. **Keep `sigma` (not `tau`), `rho12` for residual correlation. ML is
the default** (REML likelihoods aren't comparable across fixed-effect structures
— needed for model selection); REML is an option.

---

## 4. Architecture (how the engine works)

Latent: **augmented state**, node-major over the `2p−1` tree nodes × 4 axes;
data attach only at the p leaf nodes.

- **Prior precision** `P = kron(Q_topology, Λ⁻¹)` — sparse, O(p) nnz, well-
  conditioned (Hadfield–Nakagawa). *Never forms a dense Σ_phy.* → `sparse_phy.jl`,
  `prior_precision`.
- **Laplace marginal** `L(θ) = −jn(û,θ) − ½logdet H_uu + ½logdet P`. The inner
  mode û (the **E-step**) is `estep_mode` in `sparse_aug_plsm.jl`: a **fast-path
  (cheap damped Newton, accept once ‖∇u‖<1e-3) with robust trust-region LM
  fallback** (hybrid Fisher/observed Hessian, anti-collapse trust region). The
  single biggest robustness+speed lever; merged from a 4-strategy search.
- **Exact O(p) gradient** `marginal_and_exact_grad` (`fit_q4_sparse_tmb.jl`):
  cheap term + implicit-function correction; all CHOLMOD-blocked log-det
  derivatives via **Takahashi selected inversion** (`takahashi_selinv.jl`) at the
  sparse pattern. Validated to 1e-6 vs finite-diff. **This is the edge over
  gllvmTMB's O(p²) multi-trait gradient.**
- **Optimiser:** LBFGS + MoreThuente, **mean-objective** normalisation (scale-
  invariant across p), Inf barrier on PD/ρ-guard violations, **relative-objective
  stopping** (the singular variance boundary makes ‖g‖ plateau; gradient-norm
  stopping never fires — same spot where drmTMB reports "false convergence 8").
- **O(p) data sampler** (`bench/run_scaling.jl` `gen`): `u ~ N(0,P⁻¹)` via sparse
  `cholesky(P)` + triangular solve (`F.UP \ z`); verified `Cov(û)≈P⁻¹` at p=8.
  Replaces the dense-Σ_phy O(p³) generator (capped ~p=3000).

**Core files** (`src/`, wired): `sparse_phy` · `takahashi_selinv` ·
`sparse_aug_plsm` · `sparse_em_fit` (`make_problem(...; species=)` and
`mstep_beta`, both reached via `fit_ml_q4`/`gaussian_bivariate` — but **not**
`mstep_Lambda`/`fit_em_aug`: production replaces the closed-form Λ step with
`fit_ml_q4.jl`'s line-searched ascent, off the public `drm()` path; #472. The
p=100 *descent* that justified abandoning it was the **#577 sparsity-pattern
artefact** — with the pattern repaired the same step ascends +0.70 nats on the
q4_p100 fixture where it used to lose 55.8. #472 stays OPEN and fenced: a
synthetic pure-noise p=100 tree still descends, unexplained) ·
`fit_ml_q4` · `fit_q4_sparse_tmb` · the family modules (`gaussian_*`, `student`,
`poisson`, `negbinomial`, `beta`, `betabinomial`, `binomial`, `gamma`,
`lognormal`, `zeroonebeta`, `tweedie`, `cumulative`) · `inference` (Wald +
profile + bootstrap; `infer_q4` graduated here) · `summary` · `visualization` ·
`DRModels.jl`.
**Experimental** (`src/experimental/`, migrated; **not fully wired** into the
public API — do not claim otherwise): leftover `location_only` prototype copy ·
`fit_em_natgrad` (**#13 FAIL** — stalls vs sparse TMB; not a public solver) ·
`estep_{lm,trustregion,armijoguard,initprior}` · dense oracle (`q4_em_dense`,
`fit_q4_tmbgrad`) · warm-start / SQUAREM EM variants.
**Public (in `src/`, included):** opt-in REML (`reml_q4.jl`, `method = :REML`),
conjugate-EM phylo-mean (`location_only.jl`, `algorithm = :em`), and the
extracted Fisher / observed-info metric `lc_metric` (`lc_metric.jl`, #13 S1b
infra for AI-REML — **not** `algorithm = :natgrad`). Those promotions do
**not** mean the whole `experimental/` tree is wired.

---

## 5. The journey — what was built, and the dead-ends (so you don't re-walk them)

1. **Dense Laplace failed** (229 s / NaN): forgot the sparse Hadfield–Nakagawa
   precision and nested AD. → the **sparse augmented-state** route. *Lesson: never
   form dense Σ_phy; use the sparse precision + Takahashi.*
2. **The "12.5 s/eval O(p²) bug" was a myth** — a prior session *assumed* it and
   never profiled. Measured: the exact gradient is **16.5 ms**, fully O(p). The
   real blocker was **2 of 17 gradient components** (the within-response μ↔logσ
   Cholesky entries) — a **removable singularity at the ρ=0 + diagonal-Λ init**.
   Fixed by starting Λ0 off-diagonal. *Lesson: verify, don't trust prior claims.*
3. **Line search**, not start point, caused a 120-iter crawl: `BackTracking` is
   Armijo-only. **MoreThuente** (Wolfe) → the 1.86×→2.18× win.
4. **SQUAREM: tested, rejected** — it accelerates EM iteration *count*, but the
   finite-diff per-iter cost dominated (15.6 s, *slower*). The exact gradient was
   the real lever, not convergence tricks.
5. **The EM stalls on location-scale** (−259.79, suboptimal): block-coordinate
   ascent can't navigate the β↔σ coupling in the non-conjugate model. *Verified
   negative.* Joint quasi-Newton (TMB-like) is the right engine here. **But EM
   wins on the conjugate location-only model (3.1×)** — the **conjugacy thesis**,
   now measured both ways.
6. **The scale-RE degeneracy** (a 4-agent search *proved* it): with **1 obs per
   species** the inner objective is unbounded below at scale (the scale latents
   slide to σ→0). The Watanabe-singular boundary the info-geometry scout flagged.
   **Fix: nrep ≥ 2 replicates** (a modeling decision). This is statistics, not a
   solver bug — no mode-finder can find a mode that doesn't exist.
7. **Warm-start across a bootstrap grid: negative result** (0.8×) — the flat-
   region crawl dominates per-fit cost, not the approach distance.
8. **Robust mode-finder merge** then **fast-path-fallback** recovered speed
   (4.83 → 1.14 s) while keeping robustness; **multi-obs refactor** + **O(p)
   sampler** unlocked p=10,000; the **comparison suite** (location-only / REML /
   inference / grid) closed it out.

**Cross-cutting disciplines that made this work** (keep them):
- **Verify before claiming** — every headline was reproduced by an independent
  run; this caught the "12.5 s myth", the lc3/lc7 bug, and a stale auto-grid.
- **Autonomous workflows with verify-and-revert** — each optimization backed up
  the file, ran its gate, and reverted on failure, so the baseline never broke.
- **Information geometry** (scout): natural gradient ≡ Fisher scoring ≡ AI-REML;
  the variance-boundary plateau is degenerate Fisher info (Watanabe), not a bug.

---

## 6. Gotchas (read before touching the engine)

1. **lc3/lc7 removable singularity** — at ρ=0 + diagonal Λ the within-response
   μ↔logσ Cholesky gradient blows up (0/0 funnel). Start Λ0 **off-diagonal**.
   Auto-detecting this for arbitrary user data is open.
2. **Scale-RE identifiability** — use **nrep ≥ 2** obs/species, or the inner
   objective is unbounded below at scale.
3. **Singular variance boundary (Watanabe)** — near sd_phy→0, ‖g‖ plateaus and
   drmTMB's `sdreport` returns NaN. Use relative-objective stopping; for inference
   prefer bootstrap / χ̄² over Wald there.
4. **Dense-Σ_phy generator caps ~p=3000** — always use the O(p) precision sampler
   at large p.

---

## 7. Current repo state

Published, MIT, public; **v0.1.0 and v0.1.1 tagged**. `src/` core loads cleanly
(`using DRModels` resolves the include chain + exports). The `drm()` / `bf()` front
end, all 13 families (12 univariate + bivariate Gaussian), and the inference
surface (Wald + profile + bootstrap) are wired and exported; families are
validated by simulation parameter recovery (numerical drmTMB-parity gate
[#17](https://github.com/itchyshin/DRModels.jl/issues/17) is **closed**;
`DRM_PARITY_TESTS=1` stays opt-in). Several comparison-suite capabilities have
since been **promoted into the public module and exported**: opt-in **REML**
(`method = :REML`, #11/#235) with the model-selection guard, the **conjugate-EM**
Gaussian phylo-mean solver (`algorithm = :em`, from
`experimental/location_only.jl`, #12/#224), **heritability / repeatability /
ICC** accessors (#233), **DHARMa quantile residuals** (#183/#234), and the
**coevolution q=4 phylo front end** + accessors (#201/#187/#188). Julia-side
bridge helpers (`drm_bridge` / `drm_bridge_inference` + sparse-phylo companion)
exist; **#5 / Phase 1.5 is closed** (experimental bar; #349 + drmTMB #878).
`src/experimental/` is **no longer the home of the promoted features above** but
is **not fully wired** — alternative estimator prototypes remain outside the
public API (SQUAREM / natural-gradient EM, trust-region & line-search E-steps,
dense q=4 EM, warm-start fit variants, leftover `location_only` copy). There is
**no** `experimental/reml_q4.jl` on tip — public REML is `src/reml_q4.jl`.
`bench/` has runnable benchmarks + the `q4_p100` fixtures + the R fixture-gen.
`report/` has all 13 design/provenance docs. CI is Linux-only, PR +
`workflow_dispatch` (cost-disciplined).
**Honest — distribution:** S2/S3 (#340–#342) landed; **General deferred** until
readiness (**D-111**: R twin catch-up + both working; drmTMB likely CRAN first).
Do not claim membership; do not pursue Registrator. Tip idle after #357 handover;
Next = owner-opened goal — not General. **Version / tag:** `0.1.2` /
`v0.1.2` are git artifacts only (#346).

---

## 8. Roadmap to v0.1.0 (ordered)

> **Update (Phase 0, 2026-05-30):** the team ([`AGENTS.md`](AGENTS.md)), the 10
> scripted workflows (`.claude/workflows/`), the Documenter shell (`docs/`),
> `bench/Project.toml`, `TagBot.yml` / `Documenter.yml`, and the GitHub Issues
> ledger now exist. See [`ROADMAP.md`](ROADMAP.md) for the live phase plan; the
> engine steps below (A–D) remain the substantive roadmap.

**A. Package hygiene (first).** Wire a clean module (single base include, remove
load-time prints, deliberate public API); make `Pkg.instantiate(); using DRM;
Pkg.test()` green (the migrated `test/*.jl` need path/`using DRM` fixes); add
`Manifest.toml`, `docs/` Documenter site, `TagBot.yml`/`Documenter.yml`,
`bench/Project.toml`.
**B. Wire `experimental/` into the API (SCOPED).** The q=4 bivariate `bf()` front
end, `infer_q4`, public REML (`method = :REML`), conjugate-EM (`algorithm = :em`),
and `lc_metric` Fisher infra (#13 S1b) are public; **#13 natgrad solver FAIL** —
do not expose `:natgrad`. Remaining EM / E-step / dense-oracle prototypes stay
in `src/experimental/`. Bootstrap threading already has entry points (#131/#132).
**C. Open research items.** Exact REML / AI-REML gradient via `lc_metric` (#11
follow-up); χ̄² boundary inference where not yet exported; VA/ELBO (#136 open —
Experimental random-intercept path; 136e / phylo/ZI deferred). ~~drmTMB head-to-head at nrep=4/p>100~~ **done in #376** (measured;
“~12×” extrapolation retired).
**D. v0.1.0 pilot scope.** Gaussian-only, fixed-effects pilot (univariate +
bivariate distributional regression, closed-form marginal) shipped with the
bench/ADEMP harness vs drmTMB; then grow toward the phylogenetic Laplace path
that already works here.

---

## 9. Decisions

**Made:** package name `DRModels` (drops the TMB suffix, matches GLLVModels.jl pattern);
**MIT license** (Julia-ecosystem norm + matches GLLVModels.jl; DRModels.jl is fresh code,
not a port of drmTMB's GPL source, so it's legally free to be MIT); **public**
repo at `itchyshin/DRModels.jl`; cost-disciplined CI.
**Open:** whether to vendor any drmTMB GPL source later (would force GPL —
avoid, or isolate); Phase 1.5 bridge finish (#5).
**Closed this arc / deferred:** Julia General-registry submission — **not until
ready** (**D-111**); `v0.1.2` may remain as a git tag without General.

---

## 10. Provenance (`report/`, 13 docs — the full record)

`comparison-grid.md` (verified grid) · `plan-and-timings.md` (the running log:
every measurement, dead-end, fix) · `info-geometry-scout.md` (Amari/Watanabe:
natural-gradient = AI-REML; the singular boundary) · `why-q4-plsm-matters.md` ·
`q4-laplace-findings.md` · `q4-sparse-status.md` · `laplace-design.md` ·
`DRM-architecture.md` · `em-acceleration-recipe.md` ·
`drmTMB-q4-numerical-recipes.md` · `three-way-comparison.md` ·
`GLLVM-porting-playbook.md` · `summary.md`. Original scratch lives in the
`drm-julia-poc/` folder (not migrated).

---

## 11. How to verify right now

```bash
cd DRM.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); using DRModels; println("DRModels loaded")'
julia --project=. bench/run_sparse_tmb_nd.jl   # ~1.1 s, logLik −256.51, 2.18× vs drmTMB
julia --project=. bench/run_scaling.jl         # k≈1.08 to p=10,000
```

*Until step A.1 (clean module wiring), `bench/` runners may need the poc env or
small path fixes — they were validated in `drm-julia-poc/`, the migration source.*
