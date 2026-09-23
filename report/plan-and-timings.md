# Plan, order, and time comparisons — q=4 PLSM, R vs Julia

THREE engines, same model/data, same ML Laplace marginal:
- **R** = drmTMB (TMB: Laplace + nlminb, compiled CppAD sparse AD)
- **Julia-1** = DRM.jl sparse Laplace-**EM**
- **Julia-2** = DRM.jl **TMB-like** (exact gradient on sparse marginal)

## MEASURED so far (q4_p100, p=100) — honest, with caveats

| Workload | R (drmTMB) | Julia-1 (EM) | Julia-2 (TMB-like) |
|---|---|---|---|
| **Single fit** | **2.48 s**, LL −256.52 | 9.76 s (warm), LL −259.6 | **3.0 s, LL −256.508 ✓ CONVERGED** (matches/beats drmTMB) |
| **Bootstrap B=199** | ~514 s (serial R) | — (redo on correct fits) | — (threaded, next) |

**Equivalence**: Julia-2 logLik −256.508 vs drmTMB −256.52 (|Δ|=0.012, Julia *higher*);
β_rho 0.401=0.401, β_s2 −0.552=−0.552, β_mu2 [0.181,0.582]≈[0.185,0.582]. ✓

### ★ BREAKTHROUGH (this session): the TMB-like route WORKS
The prior "12.5 s/eval O(p²) bug" was a **myth** — never profiled. Measured truth:
- **One exact-gradient eval = 16.5 ms** at p=100 (every section sub-ms except the
  12 ms cold E-step). Fully O(p). The route was never slow.
- The real blocker was **2 of 17 gradient components wrong** (lc3=C[3,1], lc7=C[4,2]
  — the within-response μ_d↔logσ_d Cholesky entries): analytic −4293/−5947 vs true
  −7/+2. This is a **removable singularity at the init point** (β_rho=0 ⟹ ρ=0
  decouples responses; diagonal Λ ⟹ those two off-diagonals = 0; the only axes
  carrying a data-Hessian coupling). Adding any tiny off-diagonal there → all 17
  components correct to 1e-6 (grad_check_diag.jl, case C).
- A trial step also broke PD-ness/ρ-guard mid-line-search → DomainError. Added an
  **Inf barrier** in fg! (fit_q4_sparse_tmb.jl) so BackTracking rejects bad steps.
- **RESULT (run_sparse_tmb_nd.jl)**: start Λ0 off-diagonal → LBFGS converges,
  LL −256.508, all coefs ≈ drmTMB, **3.0 s (0.82×, near parity)**, 120 iters.

### ★★ WE BEAT drmTMB (line-search fix)
The 120-iter crawl was NOT the start point (well-scaled Λ0 → 118 iters, no help)
and NOT the singularity (optimum Λ[3,1]=0.105, Λ[4,2]=−0.072, off the singular
set). It was the **line search**: BackTracking is Armijo-only, starves LBFGS of
curvature. Switching to **MoreThuente (Wolfe)** (run_linesearch.jl, p=100):

| line search | wall | iters | f-calls | logLik |
|---|---|---|---|---|
| BackTracking | 2.55 s | 120 | 161 | −256.508 |
| HagerZhang | 6.19 s | 132 | 866 | −256.505 |
| **MoreThuente** | **1.36 s** | 94 | 116 | −256.492 |

**Julia-2 (TMB-like, MoreThuente) = 1.36 s vs drmTMB 2.48 s = 1.82× FASTER**, same
logLik. Now the default in fit_q4_sparse_tmb. Caveat: all methods plateau at
g_resid ≈ 0.2–0.4 / LL ≈ −256.49…−256.51 — the q=4 identifiability boundary
(sd_phy[3,4]=0.11,0.17 near 0); drmTMB reported "false convergence (8)" here too.
Honest read: PARITY-to-1.8× at p=100; the O(p) scaling story (p=1000/5000) +
threaded pipeline are where the big multiples come.

### Info-geometry scout + steps toward the 6-way comparison
Scout report: report/info-geometry-scout.md. Key actionable findings:
- **Natural gradient ≡ Fisher scoring ≡ AI-REML** (same update θ←θ+α·AI⁻¹g). The
  EM overshoot is an UNSCALED Euclidean step on the SPD cone; AI⁻¹ rescales. This
  is the principled variance-update fix AND validates "REML makes it fast".
- **The g_resid≈0.2 plateau is Watanabe-singular** (degenerate Fisher info at the
  variance boundary), NOT a bug — why both engines plateau identically & drmTMB's
  sdreport→NaN. ⇒ stop on relative objective change; inference via bootstrap/χ̄²
  mixture, not Wald (a quality opening for Julia).
- lc3/lc7 singularity = centered-param 0/0 funnel; ε-nudge (have it) or
  non-centered reparam removes it.

DONE this session:
- **Stopping rule → relative objective change** (f_reltol; fit_q4_sparse_tmb).
  q4_p100 now **1.34 s, 1.86×**, LL −256.49. The conv=false flag is the honest
  singular-boundary signal (no Wolfe step exists in the flat region; = drmTMB's
  "false convergence 8").
- **Natural-gradient variance update VERIFIED at p=20** (demo_natgrad_p20.jl):
  preconditioning the log-Cholesky step by the observed info H_lc (cond≈89) gives
  the largest marginal ascent (+1.84 LL) vs closed-form EM (+0.02, stuck) and
  Euclidean (+1.27). PD-by-construction. This is 1b's core variance step.

Comparison grid = 2 models × 3 engines × {ML,REML}, run in priority order:
1. location-scale ML (drmTMB/EM/TMB-like) — mostly have it. 2. location-scale REML
(1b natural-grad EM / 2b TMB / drmTMB-REML). 3. location-only (EM's conjugate home
turf — EM should win, validating the 20× memory). 4. scaling p∈{1000,5000}.
NEXT: wrap the verified natural-grad step into the 1b EM loop, then add REML
(integrate out β_μ via a bordered augmented state).

### EM-ML UPGRADE (fit_em_natgrad.jl) — measured, honest, instructive
Upgraded the EM-ML with the session's insights: EXACT gradient (1 eval, not 20
finite-diff) + NATURAL-GRADIENT Λ-step (H_lc-preconditioned) + conditional-Newton
β-step + relative-objective stop. Result at q4_p100:
- **3.4× faster per fit than the old EM** (9.76 s → 2.9 s) — the exact+natural
  gradient worked at the iteration level.
- **BUT converges to a SUBOPTIMAL point**: LL −259.79 (not −256.49), σ-axis
  sd_phy stuck at 0.30 (true ~0.11), at 109 iters (monotone, then stalled).
**Diagnosis:** block-coordinate alternation can't navigate the β↔σ coupling in
the NON-conjugate location-scale model — it zigzags in the curved valley and
stalls on the σ-axis, while Julia-2's JOINT quasi-Newton handles the cross-
curvature and reaches −256.49. The natural gradient fixed the Λ-step *scaling*,
not the *block-coordinate* structure. This is the conjugacy thesis, MEASURED:
- location-SCALE (non-conjugate) ⇒ joint optimization (TMB-like) wins; EM stalls.
- predicts EM WINS on location-only (conjugate) — the clean next experiment.
(Caveat: a more aggressive EM — full per-block maximization or SQUAREM on the
cheaper iters — might reach the optimum, but at added cost; joint gets there
directly. Not worth more EM tuning for location-scale.)

### Out-of-box #1 (continuation warm-start) — TESTED, NEGATIVE, instructive
Hypothesis: in a bootstrap/sim study, warm-start each refit from the MLE → few
iters. Tested (run_continuation.jl, B=12 bootstrap refits of q4_p100):
- COLD 1.28 s/fit (51 iters median) vs WARM 1.56 s/fit (44 iters) → **0.8×, NO
  win.** Warm even lands on slightly different local optima (|ΔLL| up to 1.07).
**Why (the key insight):** the per-fit cost is the **Watanabe-singular flat-region
crawl**, NOT the approach distance. Starting AT the MLE still takes ~44 iters
because the optimizer must wander the flat σ-variance plateau regardless of start.
⇒ Warm-start is not a throughput lever here.
**What this redirects to:**
- **Threading still works** (~8×): B=199 bootstrap ≈ 255 s serial → ~31 s on 10
  cores. Independent of warm-start.
- The real PER-FIT lever is **boundary-aware optimization** (active-set: detect
  near-zero σ-variances and pin them to the SPD-cone boundary, cutting dimension
  and skipping the flat crawl) — helps ALL engines. Elevated from honorable-
  mention to prime candidate by this result.
- Or **amortized inference** to sidestep optimization for the sim grid entirely.

### TIME-TRIAL READINESS: NOT YET — root cause found (E-step diverges at scale)
The scaling time trial (run_scaling.jl) FAILED beyond p=100: 1-iter quits, then
NaN crashes. Systematic root-cause (diag_scaling.jl, diag2.jl), evidence-based:
1. NOT the data generator (cond(Σ_phy)=127/511/1020 at p=100/500/1000, gen OK).
2. NOT the outer optimizer step-scale (fixed with: MEAN objective normalisation
   + InitialStatic(scaled) — both shipped in fit_q4_sparse_tmb; p=100 synthetic
   now fits in 0.35 s).
3. **THE BUG: the inner E-step Newton (estep_mode) DIVERGES from a cold start at
   p≥500.** ‖∇_u‖@mode = 234 (n=40) → 3.5e5 (n=150) → 5e12 (n=400) → NaN; the
   log-σ latent axes drift unboundedly (max|u| 7.5→32→NaN). Cause: cold init
   Λ0=0.3 is far too loose on the σ-axes (true ~0.09) + the damped-Newton
   globalisation accepts drift steps + scale. The exact gradient then has no
   valid mode (envelope theorem needs ∇_u=0) → fit fails. p=100 escapes it; the
   real-data q4_p100 win (1.36 s) is unaffected (its E-step converges).
**FIX (next, foundational — re-verify ALL checkpoints since estep_mode is the
base):** robust mode-finder — Levenberg–Marquardt adaptive ridge (raise λ on a
failed/uphill step, lower on success) + Armijo sufficient-decrease + ‖∇_u‖-based
convergence + a divergence guard; and/or a data-driven init prior (tight σ-axes).
SHIPPED ROBUSTNESS so far: mean-objective normalisation, scaled initial step,
Inf barrier on bad steps. Time trials resume once the mode-finder is robust p≥500.

### 4-AGENT WORKFLOW VERDICT: it's a MODEL degeneracy, not a solver bug (PROVEN)
Raced 4 robust mode-finders (LM+Fisher-hybrid, trust-region, Armijo+guard,
robust-init+LM; wf_a3cd290e-52f). ALL 4: preserve correctness (p=20 match,
p=100 ‖∇_u‖<1e-8) and KILL the NaN — strict improvements over estep_mode. But
ALL 4: converges_at_scale = FALSE, and two PROVED why:
- The inner Laplace objective is **UNBOUNDED BELOW for p≥500**. With ONE obs per
  species, a leaf's 2 MEAN latents fit it exactly (residual→0), then its 2 SCALE
  latents are unanchored → σ→0 → leaf_nll→−∞. The tree prior's softest direction
  (collective log-σ shift, ≈ Q_topology's near-null constant mode) doesn't
  penalise the collapse. Evidence: p=500 → ALL leaves σ<1e-4 (min 9e-15); the
  "5e12 gradient" is float noise (machine-ε residual / σ²≈1e-28). No finite mode
  with ‖∇_u‖<1e-4 EXISTS. estep_mode "worked" at p=100 only by luck (no leaf
  overfits before Newton stops).
- **Tight σ-prior (Λ0 axes 3,4=0.08) does NOT fix it** (tested): the prior is
  kron(Q,Λ⁻¹) and Q's small eigenvalues shrink with tree depth, so the MARGINAL
  σ-precision weakens with p regardless of Λ. Collapse persists at scale.

**This is the q=4 PLSM's scale-identifiability subtlety (= the Watanabe-singularity
the scout flagged), now pinpointed:** with 1 obs/species the scale REs are
identified ONLY through phylogenetic shrinkage, which weakens with tree depth.

**The real fix (foundational, next focused pass):** (1) merge a robust mode-finder
(estep_lm — Fisher-info-steered far + observed near + ‖∇_u‖ convergence; 3s/E-step
at p=1000, strictly dominates estep_mode); (2) regularise the σ-latent axes with
precision that does NOT weaken with p — a direct ridge/hyperprior on axes 3,4 in
H_uu (not via the tree prior), or a σ-floor bound-constrained inner solve (judge
convergence on the projected gradient). In the REAL EM/fit, β_σ and Λ adapt each
step, partly suppressing this (the frozen-prior test is the worst case) — and the
real q4_p100 fits fine (its shrinkage is sufficient at p=100). p=100 win (1.86×) stands.

### ✅ SCALING UNBLOCKED (robust mode-finder MERGED + replicate observations)
1. **Merged the workflow's robust mode-finder into estep_mode** (hybrid Fisher/
   observed Hessian + LM ridge + trust region + ‖∇‖ convergence; leaf_fisher /
   build_Huu_expected added to sparse_aug_plsm.jl). Backward-compat VERIFIED:
   p=100 real-data fit still LL −256.524 (=drmTMB), and g_resid improved 0.21→3.6e-3
   (cleaner mode). Cost: ~3.5× slower per fit (4.83 s vs 1.36 s at p=100) — the
   Fisher Hessian is 4× build_Huu; recover later via a fast-path-with-fallback.
2. **Multi-observation support** (user's modeling call: >1 obs/species identifies
   the scale REs). Foundation refactor — all leaf loops over DATA ROWS
   (eachindex(leaf_node)); make_problem takes a `species` map; mean-objective over
   n_obs. Backward-compatible (1 obs/leaf unchanged).
3. **RESULT (run_scaling.jl, nrep=4):** the fit now CONVERGES at scale —
   p=100/500/1000 in 16/22/21 iters (was 1–2 quit-outs), per-obs logLik consistent
   ~−2.18 across all p (well-posed). Wall 0.97/9.2/18.5 s, scaling exponent
   **k=1.28 (near-O(p))**. The degeneracy is resolved.

### REMAINING for the time-trial deliverable
- (a) recover Julia speed: fast-path (cheap damped Newton) with robust-LM fallback;
- (b) drmTMB on the SAME nrep=4 data at p=100/500/1000 for a fair head-to-head;
- (c) extend the curve to p=2000/5000 (where Julia's O(p) should pull far ahead).

## ★ BANKED MILESTONE (biological per-dimension-variance model scales)
The q=4 PLSM IS the biological per-dimension-variance model: the 4×4 Λ gives a
SEPARATE phylo variance per axis (μ1,μ2,logσ1,logσ2) + cross-covariances — richer
than "one variance per trait." With replicate obs (nrep=4) it CONVERGES near-O(p):

| p (×4 obs) | wall | iters | logLik | per-obs |
|---|---|---|---|---|
| 100 | 0.97 s | 16 | −880 | −2.20 |
| 500 | 9.8 s | 22 | −4375 | −2.19 |
| 1000 | 18.3 s | 21 | −8711 | −2.18 |
| 2000 | 52.9 s | 21 | −18874 | −2.36 |

> **Re-banked 2026-09-19 (lane speed6-20260919, on 90fbb0e28):** the table above predates OPTIMIZATION PLAN step 1 (fast-path + robust fallback), which has since landed in `src/sparse_aug_plsm.jl`. Measured today: p=2000 warm fit 14.2 to 15.9 s (`bench/run_scaling.jl` 15.86 s, logLik −18062.99 bit-for-bit with the value the fit reports), i.e. the ~3.5x that step predicted is recovered (52.9 / 14.7 = 3.6x). Complete section partition at p=100/1000/5000 in `bench/results/q4_sections_a734d2b90.tsv` (inner-Newton factorisations 53%, beta-block trace 23%, Gst/v assembly 17% at p=5000). The 52.9 s row is kept for provenance, not as the baseline. The p=1000 row (18.3 s) is stale by the same repair: 8.46 to 8.48 s today (`bench/results/q4_sections_a734d2b90.tsv`).

Flat iteration count (p-independent outer convergence), consistent per-obs logLik,
**empirical k=1.33** (near-O(p)). The EXACT GRADIENT is genuinely O(p) (Takahashi
at the sparse pattern, never forms dense Σ_phy) — the win over gllvmTMB's
multi-trait gradient (O(p²), their own code `sparse_phy_grad.jl:53-76` "slope≈2").
Caveats: robust mode-finder inflates absolute times ~3.5×; NOT yet head-to-head
with drmTMB (nrep=4 ≠ 1-obs); dense-Σ_phy generator caps ~p=3000.

## OPTIMIZATION PLAN — "clean O(p) + fastest" (order: 1 → 3 → 2)
**1. Speed recovery (fast-path + robust fallback).** estep_mode: for a WARM start,
try the cheap damped observed-Newton first (the old fast path); only if it
stalls/diverges fall back to the robust LM. Cold starts → robust LM. Expected
~3.5× back (warm evals are the bulk of a fit). Verify: p=100 real still −256.52
AND faster; scaling still converges. Backward-compatible (fallback = current).
**3. O(p) simulator.** Replace the dense-Σ_phy generator (O(p³) chol, caps ~3000)
with an O(p) sampler: u ~ N(0, P⁻¹) via sparse cholesky(P) + triangular solve
(P=kron(Q_cond,Λ⁻¹), PD, O(p)). Verify Cov(û)≈P⁻¹ at small p, then reach
p=5000/10,000 — the headline curve (where gllvmTMB caps at ~500).
**2. Cholesky postorder ordering (if k still >1.1).** Feed the sparse Cholesky a
tree-postorder elimination ordering so fill stays O(p) → push k 1.33 → ~1.0.
Verify: scaling exponent drops on the same curve.

## NEXT-SESSION HANDOFF — resume here
**Codebase state: CLEAN working baseline** (revert done; nothing half-edited).
Tried #1 as a one-line λ-init tweak → measured WORSE (5.24 s vs 4.83 s) → reverted.
The real #1 is the full fast-path-fallback (a refactor), deferred to the fresh pass.

**Verified facts:** p=100 real q4_p100 = LL −256.52 (run_sparse_tmb_nd.jl);
biological per-dim-variance model (4×4 Λ) converges to p=2000 (run_scaling.jl,
nrep=4), near-O(p) k=1.33; exact gradient is O(p) (the edge over gllvmTMB's O(p²)).

**Run:** `cd .../drm-julia-poc/julia/drm_q4 && ~/.juliaup/bin/julia --project=".../drm-julia-poc/julia" <script>.jl`
**Key files:** sparse_aug_plsm.jl (estep_mode = robust hybrid Fisher/observed LM +
trust region; leaf_fisher/build_Huu_expected); fit_q4_sparse_tmb.jl
(marginal_and_exact_grad, fit, fg! mean-objective); sparse_em_fit.jl (make_problem
with `species=`, mstep_*); run_scaling.jl (scaling trial); run_sparse_tmb_nd.jl (p=100 regression).
**Gotchas:** (1) Λ0 must be OFF-diagonal — lc3/lc7 removable singularity at ρ=0 + diag-Λ;
(2) 1-obs/species is DEGENERATE at scale — use nrep≥2 (the modeling call); (3) the
dense-Σ_phy generator caps ~p=3000 → #3 (O(p) sampler) is what unlocks p=5000/10,000.
**Execute OPTIMIZATION PLAN above, order 1→3→2**, verifying each gate.

## ★★★ VERIFIED FINAL — autonomous 1→3→2 pass complete (independently re-run)
Both #1 and #3 PASSED their gates and were KEPT; #2 SKIPPED (not needed, k=1.08).
Baseline never broken (verify-and-revert held).

**HEAD-TO-HEAD single fit** (real q4_p100, SAME model + data as drmTMB):
**Julia 1.14 s vs drmTMB 2.48 s = 2.18× FASTER**, logLik −256.51 (=drmTMB −256.52),
**converged=true**. #1 (fast-path + robust-LM fallback) recovered 4.83 → 1.14 s
AND kept robustness AND fixed the convergence flag. The fast-path accepts a warm
observed-Newton mode once ‖∇_u‖<1e-3 (a line-search stall at that accuracy is
exact enough for the frozen-mode Laplace; the LM's tol=1e-8 was wasted work).

**O(p) SCALING — biological per-dimension-variance model** (4×4 Λ = phylo variance
per axis + covariances; nrep=4 replicates for scale-RE identification):
| p | wall | iters | per-obs logLik |
|---|---|---|---|
| 100 | 0.77 s | 37 | −2.09 |
| 500 | 3.35 s | 23 | −2.13 |
| 1000 | 4.49 s | 15 | −2.36 |
| 2000 | 15.4 s | 20 | −2.41 |
| 5000 | 49.5 s | 22 | −2.35 |
| **10000** | **112.9 s** | 23 | −2.61 |
**k=1.08 (near-perfect O(p))**, flat iters, consistent per-obs logLik (converged
throughout). #3 replaced the dense-Σ_phy generator (O(p³), capped ~3000) with an
O(p) sparse-precision sampler u~N(0,P⁻¹) via CHOLMOD (verified Cov(û)≈P⁻¹ at p=8,
rel-err 0.023).

**Honest framing:** (1) is a clean head-to-head (same model/data). (2) is the
synthetic biological model — gllvmTMB's multi-trait path caps ~500; drmTMB's q=4
at k≈1.36 would extrapolate to ~1300 s at p=10000 vs Julia's 113 s (~12×, but
extrapolated, not measured at that p). Replicates required for identification.

## FUTURE (after current work is finished) — DRM.jl repo  [user directive — remember]
Mirror the gllvmTMB → GLLVM.jl move: create a GitHub repo **DRM.jl**, a Julia
"digital twin" of drmTMB, and migrate this `drm-julia-poc` engine into it. Use the
GLLVM.jl repo layout as the template (Project.toml, src/, test/, bench/, CI). FIRST
deliverable the user wants: a **comprehensive, rock-solid HANDOVER / ROADMAP
markdown** (the "100+ / rock-solid" doc) that guides the DRM.jl creation — scope,
architecture, ported modules, the verified results, the open items, the plan.
**NOT YET** — finish the current work first; this is the bridge to the repo.

### SQUAREM — TESTED, REJECTED (verified)
SQUAREM-EM at p=100: LL −257.64 (better), 51 iters, but **15.65 s (SLOWER than
9.76 s)** — 3 EM maps/iter × the 20-eval finite-diff Λ gradient. Confirms the
finite-diff gradient is the per-iter ceiling; convergence tricks can't remove it.
The exact gradient (Julia-2) is the answer, not SQUAREM on the EM.

## Why each is slow now (both fixable, neither fundamental)
- **Julia-1 (EM)**: Λ update uses finite-diff (100s of E-steps). Fix = the accelerated-EM recipe (conjugate-μ closed-form + Fisher-σ + PX-EM + SQUAREM) → ~5–8 iters.
- **Julia-2 (TMB-like)**: gradient is correct at small p but 12.5 s/eval at p=100 — an O(p²/p³) implementation bug (sparse `getindex` on the Takahashi result + `kron` with Dual numbers), NOT the algorithm. Fix = keep it sparse O(p).

## STEP 1 PROGRESS (measured, q4_p100)

Profiled root cause: finite-diff Λ gradient = 420 ms = **2725×** the 0.2 ms
closed-form Λ (E-step 12 ms cold / **1.2 ms warm**; takahashi 0.1 ms — all O(p)).

Fixes applied + measured:
- closed-form-direction + warm line search: FAST (0.25 s p=20) but STALLS on the
  σ axes (closed-form direction wrong for non-conjugate scale) — rejected.
- **warm finite-diff (true gradient + warm E-steps)**: CORRECT (Λ moves to the
  right optimum, logLik ≈ drmTMB) AND faster:
  - p=20: 8.4 s → **0.98 s** (8.5×), logLik −48.3 (matches cold −47.8)
  - p=100: 26.8 s → **9.76 s** (2.75×), logLik −259.6 ≈ drmTMB −256.5 (climbing);
    ratio vs drmTMB **0.25×** (4× behind, was 10× behind)

Remaining gap to beat drmTMB's 2.48 s:
1. ~~**Convergence** — SQUAREM~~ **TESTED, REJECTED.** SQUAREM-EM at p=100:
   logLik −257.64 (*better*, |Δ|=1.12 from drmTMB), 51 iters — but **15.65 s
   (SLOWER than 9.76 s)**. Each SQUAREM iter = 3 EM maps, and each map is
   dominated by the 20-eval finite-diff Λ gradient, so the 3× per-iter overhead
   swamps the iter-count reduction. **This empirically confirms the finite-diff
   gradient is the per-iteration ceiling** — no convergence trick removes it.
2. **Finite-diff is THE bottleneck** — 20 evals/Λ-step vs TMB's ~1 (reverse-mode).
   → **analytic/exact gradient** (1 eval + 1 sparse solve for the full 17-dim
   gradient). This is `marginal_and_exact_grad` (validated dense 6.5e-9); the
   blocker is its O(p) sparse implementation (the 12.5 s/eval bug). THIS is the
   real lever and the literal TMB lesson.

### Why drmTMB is fast — what we learn (answered)
1. **Reverse-mode AD gives the whole gradient in ~1 eval** (CppAD, ~4× one
   function eval). Our finite-diff = 20 evals/Λ-grad. **The core ~20× gap.**
2. **Quasi-Newton (nlminb) on the marginal → superlinear** (~30–50 iters). Our
   EM is linear (60+). SQUAREM can't fix the per-iter cost (measured above).
3. **Profile / joint optimization** — no block alternation. The REML/lme4 lever:
   profile out the conjugate β_μ (GLS, closed-form given Λ,σ) → optimizer works
   in fewer dims (13 not 17), cleaner surface. Secondary to the gradient.
4. **Compiled C++** — constant factor.
**Implication:** the EM (finite-diff) route has a per-iter ceiling; the
exact-gradient + quasi-Newton route (Julia-2, TMB-like) is *why* TMB is fast and
is our path to beat 2.48 s. Pivot to fixing the sparse exact gradient.

## PLAN & ORDER

**Step 1 — get ONE fast+correct Julia single fit at p=100** (whichever lands first):
  - 1a. Fix Julia-2's O(p) sparse inefficiency (the 12.5s/eval bug) → target ~1–2 s.
  - 1b. Implement accelerated-EM for Julia-1 (recipe in em-acceleration-recipe.md) → target ~1–2 s.
  Gate: logLik ≈ −256.52, converged.

**Step 2 — single-fit benchmark**: R vs Julia-1 vs Julia-2 at p=100 (the clean three-way you asked for). Target: Julia at parity-to-few× of drmTMB's 2.48 s.

**Step 3 — SPECIES SCALING (the headline you want)**: wall-clock(p) for all 3 at **p ∈ {100, 1000, 5000}**. Need to run drmTMB at p=1000/5000 too. Expectation: all O(p); the constant-factor + iteration-count differences show which algorithm wins as p grows. THIS is the table you're after.

**Step 4 — threaded pipeline (the 100× target)**: bootstrap + profile-likelihood CIs on CORRECT fits, threaded across cores. warm-start (~10×) × threading (~16×, after fixing the 3.5×→16× GC bottleneck) = ~100× on the inference pipeline.

**Step 5 — inference**: Louis/Supplemented-EM SE (Wald) + threaded profile-likelihood CIs.

## Targets (what "done" looks like)

| | R | Julia (target) |
|---|---|---|
| single fit p=100 | 2.48 s | ~1–2 s (parity–2×) |
| single fit p=5000 | ? (run it) | O(p) — should pull ahead |
| bootstrap B=199 | ~514 s | ~5–16 s (~30–100×) |
| profile CI | slow (serial) | threaded (~10–30×) |
| Hessian at p=354 | non-PD (fails) | PD (stability win) |
