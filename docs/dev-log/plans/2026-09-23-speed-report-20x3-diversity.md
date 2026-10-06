# Speed report — aim ~20 cells × 3 packages (diversity plan)

**Status:** Floor **DONE** at **10 / 10 / 12** (`has_receipt`). Aim **~20 each** — **not claimed until banked**.
**Authority:** `2026-09-23-three-package-speed-board.md` (32-cell matrix) · Totoro standing order · D-50.
**This file:** taxonomy + expansion matrix + report outline + first-wave ranking. **No Totoro launch from this plan alone.**

Active lenses: Shannon, Ada, Rose (perspectives). Spawned subagents: none.

---

## 0. Hard fences (Rose)

1. **Do not claim “20/20/20”** (or any count above banked) until every quoted cell is `has_receipt` with dated TSV + host/threads/BLAS + SHA.
2. **Paired × only when R (or Julia-before) measured on the same host/threads/DGP.** Absolute tip walls are fine; never invent a speedup from abs alone.
3. **Not all models.** Reader report names coverage as a *sample of fittable DGPs*, not the capability ledger.
4. **Never cross-host ×.** Mac Studio vs Totoro EPYC stay separate columns.
5. **ASReml / Latte / GMRF** appear only as labelled comparators when receipts exist; adoption stays out of scope.
6. Public README/NEWS speed wording stays **withheld** until Rose signs a report draft that cites cell_ids only.

---

## 1. Taxonomy — “kinds of data” per package

Kinds are **DGP / model-structure labels** for diversity scoring. A cell covers a kind if its DGP exercises that structure (one cell may cover two kinds).

### 1.1 GLLVModels (`GLLVM.jl`)

| Kind ID | Kind | Fittable today? | Board coverage (floor) |
|---|---|---|---|
| `G-gauss-unstruct` | Gaussian unstructured LV / closed-form | yes | `gllvm-gauss-unstruct-{small,large}` |
| `G-gauss-phylo-em` | Gaussian structured phylo **EM** (sparse E-step) | yes (EM path; not default public fitter) | `gllvm-gauss-phylo-em-p{200,1000,5000}` |
| `G-nongauss-glmm` | Non-Gaussian grouped GLMM (Poisson Laplace) | yes | `gllvm-pois-glmm-{200x5,5000x3}` + hess |
| `G-nb-binom-lv` | NB2 / Binomial unstructured Laplace LV | yes | `gllvm-nb2-or-binom-unstruct` |
| `G-profile-ci` | Post-fit profile CI wall | yes (small) | `gllvm-profile-ci-small` |
| `G-gauss-lv-large` | Larger Gaussian LV (published / T4 scale) | yes | *gap vs board IDs* (T4 exists historically; not in 10-cell board IDs) |
| `G-binom-glmm` | Binomial grouped GLMM fit wall | yes (identity covered #448; **wall unmeasured**) | **gap** |
| `G-nb2-lv-scale` | NB2 LV at larger (p,n) | yes | **gap** (T4 NB2 97.9 s historical) |
| `G-phylo-nongauss` | Non-Gaussian phylo (Laplace / AGHQ) | partial / route-gated | **gap** — mark `blocked` if route not tip-ready |
| `G-spatial` | SPDE / spatial latent | partial | **gap** — only if tip route fittable without engine surgery |
| `G-animal` | Animal / pedigree structured | partial | **gap** — same gate |
| `G-ordinal-or-zi` | Ordinal / zero-inflated LV | partial | **later wave** (diversity; not first-wave unless smoke green) |
| `G-latte-gap` | Same fixture vs Latte.jl | comparator only | optional labelled row; timing after identity |

### 1.2 DRModels (`DRM.jl`)

| Kind ID | Kind | Fittable today? | Board coverage (floor) |
|---|---|---|---|
| `D-gauss-q4-phylo` | Gaussian q=4 phylo ML scale ladder | yes | `drm-gauss-q4-phylo-p{100,1000,5000}` |
| `D-gauss-locscale` | Gaussian unstructured location-scale | yes | `drm-gauss-locscale-n1000` + bridge |
| `D-relmat` | Structured `relmat` | yes | `drm-gauss-relmat-G25` |
| `D-meta-V` | Known sampling cov `meta_V` | yes | `drm-bridge-meta-V` |
| `D-bridge-nongauss` | Bridge NB2 / Poisson | yes | `drm-bridge-{nbinom2,poisson}` |
| `D-crossed-pois` | Crossed RE Poisson | yes | `drm-crossed-poisson` |
| `D-phylo-pois` | Phylo Poisson | yes | `drm-phylo-poisson` (Totoro Wave B) |
| `D-phylo-nb2` | Phylo NB2 | yes | `drm-phylo-nb2` |
| `D-phylo-binom` | Phylo binomial | yes | `drm-phylo-binomial` |
| `D-phylo-gamma-beta` | Phylo Gamma / Beta | yes (Julia-only where TMB cannot) | `drm-phylo-gamma` + `drm-phylo-beta-p128` |
| `D-crossed-family` | Crossed RE other families | yes | `drm-crossed-binomial` |
| `D-biv-gauss` | Bivariate Gaussian / `rho12` | yes | `drm-biv-gauss-rho12` (+ bridge Wave A) |
| `D-animal` | `animal()` pedigree | yes if tip admits | `drm-animal-gauss` |
| `D-spatial` | Spatial / mesh | route-gated | **gap** / `blocked` if not tip-ready |
| `D-lss` | Location-scale-scale / `sd() ~ x` | yes (narrow) | `drm-lss-sd-slope` |
| `D-profile-ci` | Profile CI wall | yes | `drm-profile-ci-locscale` |
| `D-large-n` | Large-n / large-p abs wall (beyond q4 p5k) | yes | **gap** (e.g. crossed n20k already in notes; promote or new) |
| `D-reml` | REML vs ML wall | yes | **later** (estimator confound; label carefully) |

### 1.3 HSquared.jl

| Kind ID | Kind | Fittable today? | Board coverage (floor) |
|---|---|---|---|
| `H-animal-scale` | Gaussian animal AI-REML q-ladder | yes | `hsq-animal-fit-q{500,2000,10000,20000}` |
| `H-aireml-iter` | AI-REML per-iter workspace | yes | `hsq-aireml-iter-workspace` |
| `H-reml-eval` | Single REML loglik eval | yes | `hsq-reml-eval-once` |
| `H-postfit` | Post-fit uncertainty / multi-effect | yes | `hsq-postfit-*` + `hsq-multi-effect-K2-q500` |
| `H-selinv` | Selected-inverse kernel (high fill) | yes | `hsq-selinv-fill471-{kernel,simd}` |
| `H-pev` | PEV / reliability via selinv | yes | `hsq-pev-reliability-q500` |
| `H-genomic-G` | Genomic relationship / GBLUP-class | yes (phase2 / sparse AI-REML) | **gap** |
| `H-supplied-K` | Supplied `K` / custom relatedness | yes | **gap** |
| `H-maternal` | Direct–maternal | yes (narrow) | **gap** |
| `H-multivar` | Multivariate REML | yes (narrow) | **gap** |
| `H-repeat` | Repeatability / permanent env | yes | **gap** |
| `H-nongauss` | Binomial / Bernoulli / Gamma animal-ish | experimental / partial | **later** or fenced abs |
| `H-asreml` | Paired ASReml wall | gated script | **not first-wave** until Rose wording + paired receipt |
| `H-halfsib` | Benign halfsib (must not regress) | yes | optional negative-control abs |

### 1.4 Diversity score (how we know the aim is diverse)

For each package, the ~20-cell aim should hit **≥8 distinct kind IDs** from that package’s table (floor already hits ~5–7). First wave prioritizes **empty kinds**, not denser ladders of an already-covered kind (except one intentional large-n rung).

---

## 2. Cell matrix — existing 32 + proposed to ~20 each

Columns: `cell_id | package | kind(s) | DGP (short) | metric | Totoro script | R comparator | status`

### 2.1 GLLVModels — existing 10 (`has_receipt`)

| cell_id | kind(s) | DGP | metric | Totoro script | R cmp | status |
|---|---|---|---|---|---|---|
| `gllvm-gauss-unstruct-small` | G-gauss-unstruct | (p,n,K)=(8,40,1) Gaussian LV | abs fit wall | `bench/speed_bench.jl` | no (abs) | `has_receipt` |
| `gllvm-gauss-unstruct-large` | G-gauss-unstruct | (30,100,2) Gaussian LV | abs fit wall | `bench/speed_bench.jl` | no | `has_receipt` |
| `gllvm-gauss-phylo-em-p200` | G-gauss-phylo-em | phylo EM p=200 | ms/iter before→after | `bench/profile_em_phylo_scaling.jl` | no | `has_receipt` |
| `gllvm-gauss-phylo-em-p1000` | G-gauss-phylo-em | phylo EM p=1k | ms/iter × | same | no | `has_receipt` |
| `gllvm-gauss-phylo-em-p5000` | G-gauss-phylo-em | phylo EM p=5k | ms/iter × (~460×) | same | no | `has_receipt` |
| `gllvm-pois-glmm-200x5` | G-nongauss-glmm | grouped Poisson N=1000 G=200 | fit wall × | `bench/profile_grouped_glmm.jl` | no (Latte labelled separate) | `has_receipt` |
| `gllvm-pois-glmm-5000x3` | G-nongauss-glmm | `glmm_5000x3_g500` | fit wall × (#446) | same | no | `has_receipt` |
| `gllvm-pois-glmm-200x5-hess` | G-nongauss-glmm | Hessian path only | hess wall × | same | no | `has_receipt` |
| `gllvm-nb2-or-binom-unstruct` | G-nb-binom-lv | NB/Binom 8×40×1 analytic vs finite | × vs :finite | `bench/speed_bench.jl` | no | `has_receipt` |
| `gllvm-profile-ci-small` | G-profile-ci | profile CI beta[1] on count | abs CI wall | `bench/speed_bench.jl` `PROFILE_CI=1` | no | `has_receipt` |

### 2.2 GLLVModels — proposed new (aim +10 → ~20)

| cell_id | kind(s) | DGP | metric | Totoro script | R cmp | status |
|---|---|---|---|---|---|---|
| `gllvm-gauss-lv-t4-p20n500` | G-gauss-lv-large | T4 Gaussian p=20 n=500 K=2 | abs + optional vs gllvmTMB | `bench/speed_bench.jl` or T4 recipe | **yes** if R re-run | `proposed` |
| `gllvm-pois-lv-t4-p20n500` | G-nb-binom-lv / LV scale | T4 Poisson p=20 n=500 | abs (+ R if measured) | T4 / `speed_bench` large | optional | `proposed` |
| `gllvm-nb2-lv-t4-p20n500` | G-nb2-lv-scale | T4 NB2 p=20 n=500 | abs | same | optional | `proposed` |
| `gllvm-binom-glmm-200x5` | G-binom-glmm | grouped Binomial `glmm_200x5`-class | abs fit wall (shipped HESS) | `bench/profile_grouped_glmm.jl` | no | `proposed` |
| `gllvm-profile-ci-glmm` | G-profile-ci | profile on grouped Poisson small | abs | extend `profile_grouped_glmm` / confint harness | no | `proposed` |
| `gllvm-gauss-phylo-fit-p200` | G-gauss-phylo-em adjacent | non-EM sparse phylo Gaussian fit wall p=200 | abs (label ≠ EM) | `bench/sparse_phy_bench.jl` / fit path | no | `proposed` |
| `gllvm-pois-phylo-small` | G-phylo-nongauss | Poisson phylo small tip smoke | abs or `blocked` | phylo nongauss harness if tip-ready | no | `proposed` / may `blocked` |
| `gllvm-spatial-gauss-small` | G-spatial | spatial Gaussian small mesh | abs or `blocked` | SPDE fit smoke if tip-ready | no | `proposed` / may `blocked` |
| `gllvm-latte-gap-200x5` | G-latte-gap | same fixture vs Latte wall | labelled ratio (not “our ×”) | `profile_grouped_glmm` + Latte | Latte yes | `proposed` (after identity) |
| `gllvm-gauss-unstruct-p50n2k` | G-gauss-lv-large | (50,2000,2) scale | abs | `speed_bench` / T4 | optional | `proposed` |
| `gllvm-ordinal-lv-smoke` | G-ordinal-or-zi | ordinal LV tiny | abs / later | family smoke | no | `proposed-later` |
| `gllvm-animal-gauss-small` | G-animal | animal structured Gaussian | abs or `blocked` | if tip admits | no | `proposed` / may `blocked` |

### 2.3 DRModels — existing 10 (`has_receipt`)

| cell_id | kind(s) | DGP | metric | Totoro script | R cmp | status |
|---|---|---|---|---|---|---|
| `drm-gauss-q4-phylo-p100` | D-gauss-q4-phylo | q4 phylo p=100 | Julia-vs-Julia wall (no gain) | `bench/profile_q4_sections.jl` | no | `has_receipt` |
| `drm-gauss-q4-phylo-p1000` | D-gauss-q4-phylo | q4 phylo p=1k | same | same | no | `has_receipt` |
| `drm-gauss-q4-phylo-p5000` | D-gauss-q4-phylo | q4 phylo p=5k | paired + tip abs | same | no | `has_receipt` |
| `drm-gauss-locscale-n1000` | D-gauss-locscale | loc-scale n=1000 | × vs drmTMB | H2H re-anchor evidence | **yes** | `has_receipt` |
| `drm-gauss-relmat-G25` | D-relmat | relmat G=25 | Julia abs + Δll (TMB wall fenced) | H2H re-anchor | yes* fenced | `has_receipt` |
| `drm-bridge-gauss-locscale` | D-gauss-locscale | bridge n=180 | × vs TMB | `bench/bridge_six_cell_timing.jl` | **yes** | `has_receipt` |
| `drm-bridge-nbinom2` | D-bridge-nongauss | NB2 bridge | × vs TMB | same | **yes** | `has_receipt` |
| `drm-bridge-poisson` | D-bridge-nongauss | Poisson bridge | × vs TMB | same | **yes** | `has_receipt` |
| `drm-bridge-meta-V` | D-meta-V | meta_V bridge | × vs TMB | same | **yes** | `has_receipt` |
| `drm-crossed-poisson` | D-crossed-pois | crossed RE Poisson | tip abs | `bench/fit_crossed_poisson.jl` | no | `has_receipt` |

### 2.4 DRModels — proposed new (aim +10 → ~20)

| cell_id | kind(s) | DGP | metric | Totoro script | R cmp | status |
|---|---|---|---|---|---|---|
| `drm-phylo-poisson` | D-phylo-pois | phylo Poisson tip | abs (+ R if drmTMB fits) | Totoro `run_first_wave.jl` | optional | `has_receipt` (Totoro `e9d50a110`; Julia abs) |
| `drm-phylo-nb2` | D-phylo-nb2 | phylo NB2 | abs / × | same | optional | `has_receipt` |
| `drm-phylo-binomial` | D-phylo-binom | phylo binomial | abs / × | same | optional | `has_receipt` |
| `drm-phylo-gamma` | D-phylo-gamma-beta | phylo Gamma | abs; **Julia-only fence** if TMB cannot | same | no if TMB cannot | `has_receipt` |
| `drm-phylo-beta` | D-phylo-gamma-beta | phylo Beta | abs; Julia-only fence | Mac Wave A `drm-phylo-beta-p128` | no if TMB cannot | `has_receipt` |
| `drm-crossed-binomial` | D-crossed-family | crossed binomial | abs | Totoro first-wave | no | `has_receipt` |
| `drm-biv-gauss-rho12` | D-biv-gauss | bivariate Gaussian residual rho12 | abs (+ R optional) | Totoro first-wave | optional | `has_receipt` |
| `drm-animal-gauss` | D-animal | animal() Gaussian | abs / × | Totoro first-wave | optional | `has_receipt` |
| `drm-lss-sd-slope` | D-lss | `sd(group) ~ x` Gaussian | abs | Totoro first-wave | no | `has_receipt` |
| `drm-profile-ci-locscale` | D-profile-ci | profile CI on loc-scale | abs | Totoro first-wave | no | `has_receipt` |
| `drm-crossed-poisson-n20k` | D-large-n | crossed large (promote tip note) | abs | `fit_crossed_poisson.jl` fixedq_n20k | no | `proposed` |
| `drm-spatial-gauss-small` | D-spatial | spatial Gaussian small | abs or `blocked` | if tip admits | no | `proposed` / may `blocked` |
| `drm-h2h-q4-vs-tmb-p1000` | D-gauss-q4-phylo | q4 phylo vs drmTMB | × vs TMB | `bench/head_to_head_q4_scaling.jl` | **yes** | `has_receipt` (Julia abs; TMB pair owed) |

### 2.5 HSquared — existing 12 (`has_receipt`)

| cell_id | kind(s) | DGP | metric | Totoro script | R cmp | status |
|---|---|---|---|---|---|---|
| `hsq-animal-fit-q500` | H-animal-scale | animal q≈500 | wall (soft hist × fenced) | `sim/e2e_wall_receipts.jl` | no | `has_receipt` |
| `hsq-animal-fit-q2000` | H-animal-scale | q≈2k | same | same | no | `has_receipt` |
| `hsq-animal-fit-q10000` | H-animal-scale | q≈10k | abs Mac+Totoro | same | no | `has_receipt` |
| `hsq-animal-fit-q20000-large` | H-animal-scale | q≈20k | abs Mac+Totoro | same | no | `has_receipt` |
| `hsq-aireml-iter-workspace` | H-aireml-iter | AI-REML iter | × (#371) | `sim/profile_ai_reml_sections.jl` | no | `has_receipt` |
| `hsq-reml-eval-once` | H-reml-eval | one reml eval | × (#370) | #370 body / sim | no | `has_receipt` |
| `hsq-postfit-uncertainty-3call` | H-postfit | 3 post-fit calls | × | #370 | no | `has_receipt` |
| `hsq-postfit-multi-effect` | H-postfit | multi_effect_uncertainty | abs | #370 | no | `has_receipt` |
| `hsq-selinv-fill471-kernel` | H-selinv | selinv fill≈471 q=20k | × 28 | `bench/selinv_arms.jl` | no | `has_receipt` |
| `hsq-selinv-fill471-simd` | H-selinv | +SIMD | × 76 | same | no | `has_receipt` |
| `hsq-pev-reliability-q500` | H-pev | PEV q=500 | × fenced | e2e TSV | no | `has_receipt` |
| `hsq-multi-effect-K2-q500` | H-postfit | multi-effect K=2 | × estimator-confound fenced | e2e TSV | no | `has_receipt` |

### 2.6 HSquared — proposed new (aim +8 → ~20)

| cell_id | kind(s) | DGP | metric | Totoro script | R cmp | status |
|---|---|---|---|---|---|---|
| `hsq-genomic-gblup-q2k` | H-genomic-G | genomic / GBLUP-class q~2k | abs fit wall | `sim/phase5_sparse_aireml_benchmark.jl` / phase2 genomic | optional sommer | `proposed` |
| `hsq-genomic-gblup-q10k` | H-genomic-G | genomic larger | abs | same / DRAC only if array | optional | `proposed` |
| `hsq-supplied-K-q500` | H-supplied-K | supplied K animal-scale | abs | `sim/phase3_supplied_k_recovery_gate.jl` timed wrap | no | `proposed` |
| `hsq-maternal-q500` | H-maternal | direct–maternal | abs | phase4 maternal gate timed | no | `proposed` |
| `hsq-multivar-K2` | H-multivar | bivariate / MV REML small | abs | `sim/phase4_multivariate_reml_recovery.jl` timed | no | `proposed` |
| `hsq-repeatability-q500` | H-repeat | repeatability | abs | `sim/phase3_repeatability_recovery_gate.jl` timed | no | `proposed` |
| `hsq-animal-fit-q500-totoro` | H-animal-scale | re-time q500 on Totoro | abs (no cross-host ×) | `e2e_wall_receipts.jl` | no | `proposed` |
| `hsq-pev-q2000` | H-pev | PEV at q=2k | abs | e2e / selinv path | no | `proposed` |
| `hsq-halfsib-control` | H-halfsib | benign halfsib abs | abs (must not “win” via wrong path) | e2e / cpu_fit | no | `proposed` |
| `hsq-asreml-animal-q2k` | H-asreml | paired ASReml | × **only if** paired receipt | `sim/phase_s6_asreml_wallclock_ladder.jl` | **yes** ASReml | `proposed-gated` |
| `hsq-binom-animal-smoke` | H-nongauss | binomial animal-ish | abs fenced experimental | phase6 binomial timed | no | `proposed-later` |

### 2.7 Count ledger (honest)

| Package | Banked now | Proposed new (incl. gated/later) | Aim banked | Claimable today |
|---|---:|---:|---:|---|
| GLLVModels | 10 | 12 (2 may block; 1 later) | ~20 | **10** |
| DRModels | **28** | floor 10 + Wave A 8 + Wave B 10 (`e9d50a110`) | ~20 | **28** (aim cleared; soft A3 skipped) |
| HSquared | **20** | 11 planned; **8 banked** Phase B first-wave (`fdc43845`; ASReml skipped) | ~20 | **20** (floor 12 + kinds 8) |
| **Total** | **58** | — | **~60** | **58** (GLLVM still 10) |

**Claim line (updated 2026-09-23 Phase B DRM Wave B):** “Speed board: 10 + **28** + **20** attested cells. GLLVM expansion toward ~20 still in progress; do not claim 20/20/20. DRM Totoro walls are absolute (TMB pairs mostly owed).”

---

## 3. Report outline (one reader-facing markdown → HTML)

Single artifact (suggested path after first banked wave):
`docs/dev-log/reports/2026-09-speed-three-engines.md` (+ optional `pkgdown`/Documenter page later — **not** this slice).

### 3.1 Structure

1. **Title + one-sentence purpose** — measured fit / post-fit / kernel walls on three Julia engines; sample of DGPs, not full capability.
2. **How to read** — columns: abs wall vs paired ×; host/threads; SHA; what “R comparator” means.
3. **Honest fences box** (fixed copy):
   - Not every model class.
   - × only when paired on same machine.
   - EM / kernel / post-fit labelled separately from end-to-end user fit.
   - DRM q4 Julia-vs-Julia: attested **no wall gain** after speed6.
   - H² selinv kernel ≠ ASReml claim.
   - GLLVM phylo EM ≠ default public fitter.
4. **Per-package section** (GLLVModels → DRModels → HSquared):
   - 1 paragraph “what sped up / what did not”
   - Table of **banked** cells only (cell_id, kind, metric, wall/×, receipt link)
   - “Kinds still missing” bullet list from taxonomy gaps
5. **Cross-package diversity map** — kind coverage heatmap (kinds × package: banked / proposed / blocked).
6. **Comparator appendix** — drmTMB / gllvmTMB / Latte / ASReml rows only with receipts; else “not measured”.
7. **Methods** — Totoro pin, thread pins, identity-before-timing rule, TSV naming `board_<pkg>_<yyyymmdd>_<sha>.tsv`.
8. **Open work** — first-wave queue; blocked kinds; no claim of 20 until banked.

### 3.2 What the report must not do

- Headline a single × across packages.
- Quote historical T4 / Aug-24 numbers without re-anchor SHA when presented as “current tip”.
- Promote blocked spatial/animal cells as measured.

---

## 4. First wave — rank ≤10 new cells per package

Measure **empty kinds first**, scripts that already exist, Totoro CPU (≤100 cores). Skip `blocked` / ASReml-gated until tip-ready + Rose.

### 4.1 GLLVModels — first 10 new (priority order)

| Rank | cell_id | Why first |
|---:|---|---|
| 1 | `gllvm-binom-glmm-200x5` | Kind gap; identity already green (#448); cheap grouped wall |
| 2 | `gllvm-gauss-lv-t4-p20n500` | Reader-facing LV scale; optional R pair |
| 3 | `gllvm-pois-lv-t4-p20n500` | Non-Gaussian LV abs (not only GLMM) |
| 4 | `gllvm-nb2-lv-t4-p20n500` | Slow path diversity (historical 97.9 s class) |
| 5 | `gllvm-gauss-unstruct-p50n2k` | Large-n unstructured without new engine |
| 6 | `gllvm-gauss-phylo-fit-p200` | Distinguish **fit** wall from EM ms/iter |
| 7 | `gllvm-profile-ci-glmm` | Profile beyond tiny unstructured |
| 8 | `gllvm-latte-gap-200x5` | Labelled comparator (after identity gate) |
| 9 | `gllvm-pois-phylo-small` | Or `blocked` after 30‑min tip smoke |
| 10 | `gllvm-spatial-gauss-small` | Or `blocked`; do not force engine work |

Reserve: `gllvm-animal-gauss-small`, `gllvm-ordinal-lv-smoke`.

### 4.2 DRModels — first 10 new (priority order)

**BANKED 2026-09-23 Totoro Wave B** (`e9d50a110`, threads=1; Julia abs; soft A3 skipped).
Receipt: `DRModels.jl` `docs/dev-log/evidence/2026-09-23-speed-kinds-toward20/board_drm_first_wave_20260923_e9d50a110.csv`.

| Rank | cell_id | Why first | status |
|---:|---|---|---|
| 1 | `drm-phylo-poisson` | Non-Gaussian phylo kind empty; script exists | `has_receipt` |
| 2 | `drm-phylo-nb2` | Same family | `has_receipt` |
| 3 | `drm-phylo-binomial` | Same family | `has_receipt` |
| 4 | `drm-h2h-q4-vs-tmb-p1000` | Paired × vs TMB on q4 (complements Julia-vs-Julia no-gain) | `has_receipt` (Julia abs; TMB owed) |
| 5 | `drm-phylo-gamma` | Julia-only honest fence (where TMB cannot) | `has_receipt` |
| 6 | `drm-crossed-binomial` | Crossed beyond Poisson | `has_receipt` |
| 7 | `drm-biv-gauss-rho12` | Bivariate kind empty | `has_receipt` |
| 8 | `drm-profile-ci-locscale` | Inference wall diversity | `has_receipt` |
| 9 | `drm-animal-gauss` | Pedigree kind | `has_receipt` |
| 10 | `drm-lss-sd-slope` | Loc-scale-scale kind | `has_receipt` |

Reserve: `drm-phylo-beta` (**banked** as Wave A `drm-phylo-beta-p128`), `drm-crossed-poisson-n20k` (promote existing tip), `drm-spatial-gauss-small` (may block).

### 4.3 HSquared — first 10 new (priority order)

| Rank | cell_id | Why first |
|---:|---|---|
| 1 | `hsq-genomic-gblup-q2k` | Genomic kind empty; core product diversity |
| 2 | `hsq-supplied-K-q500` | Custom relatedness kind |
| 3 | `hsq-maternal-q500` | Maternal kind |
| 4 | `hsq-repeatability-q500` | Repeatability kind |
| 5 | `hsq-multivar-K2` | Multivariate kind |
| 6 | `hsq-animal-fit-q500-totoro` | Same DGP Totoro abs (no Mac×Totoro) |
| 7 | `hsq-pev-q2000` | PEV scale beyond q500 |
| 8 | `hsq-genomic-gblup-q10k` | Large-n genomic |
| 9 | `hsq-halfsib-control` | Benign control (anti-false-win) |
| 10 | `hsq-asreml-animal-q2k` | **Gated** — only if ASReml available + Rose wording ready; else skip and stop at 9 |

Reserve: `hsq-binom-animal-smoke` (experimental fence).

---

## 5. Execution notes (when G0 unlocks measurement)

1. Prefer **one Totoro batch per package** writing `bench/results/board_<pkg>_2026XXXX_<sha>.tsv` (or `docs/dev-log/evidence/…`).
2. Flip status `proposed` → `has_receipt` only with SHA + host + threads + metric.
3. If a tip smoke fails in ≤30 min → mark `blocked` in the board; do not hold the report for engine work.
4. After each package banks +5 new kinds-covering cells, refresh §2 count ledger and regenerate report tables from receipts only.
5. Sibling speed arcs (Latte default-ON, DRM beta-trace, H² SelectedInversion wire) stay **orthogonal** — they may produce walls usable as cells, but this plan does not authorize `src/` changes.

---

## 6. Return summary (for orchestrator)

| Item | Value |
|---|---|
| Plan path | `~/local-scratch/lanes/GLLVM.jl-s9cov-20260921/docs/dev-log/plans/2026-09-23-speed-report-20x3-diversity.md` |
| Banked floor | 10 / 10 / 12 — **DONE** |
| Aim | ~20 / ~20 / ~20 — **not claimed** |
| First-wave GLLVM (10) | `gllvm-binom-glmm-200x5`, `gllvm-gauss-lv-t4-p20n500`, `gllvm-pois-lv-t4-p20n500`, `gllvm-nb2-lv-t4-p20n500`, `gllvm-gauss-unstruct-p50n2k`, `gllvm-gauss-phylo-fit-p200`, `gllvm-profile-ci-glmm`, `gllvm-latte-gap-200x5`, `gllvm-pois-phylo-small`, `gllvm-spatial-gauss-small` |
| First-wave DRM (10) | `drm-phylo-poisson`, `drm-phylo-nb2`, `drm-phylo-binomial`, `drm-h2h-q4-vs-tmb-p1000`, `drm-phylo-gamma`, `drm-crossed-binomial`, `drm-biv-gauss-rho12`, `drm-profile-ci-locscale`, `drm-animal-gauss`, `drm-lss-sd-slope` |
| First-wave H² (10) | `hsq-genomic-gblup-q2k`, `hsq-supplied-K-q500`, `hsq-maternal-q500`, `hsq-repeatability-q500`, `hsq-multivar-K2`, `hsq-animal-fit-q500-totoro`, `hsq-pev-q2000`, `hsq-genomic-gblup-q10k`, `hsq-halfsib-control`, `hsq-asreml-animal-q2k` (gated) |

**Status:** Plan written. Measurement **PAUSED** until G0 / compute unlock. Do not claim 20.
