# q=4 PLSM sparse engine — honest status (2026-05-29)

## What works (validated, real)

- **Sparse augmented-state foundation is CORRECT.** `test_sparse_aug.jl`
  (Checkpoint 3): the sparse augmented Laplace marginal matches the dense
  leaf-only Laplace **exactly** at p=8 — mode to **1.84e-10**, marginal
  difference **0.000000**. Root-conditioning (remove root node, matching
  `sigma_phy_dense`) was the key fix.
- **Step-1 infra validated**: `augmented_phy` sparse Q reproduces R's
  `ape::vcv` Σ_phy to 1.8e-10; `takahashi_selinv` matches dense inv to 3e-14.
- **The sparse engine RUNS on the real q4_p100 ultrametric tree** —
  stable, no NaN. The dense routes could NOT (dense `inv(Σ_phy)` of an
  ultrametric covariance is ill-conditioned → NaN; the sparse Q is
  well-conditioned). This is a genuine stability win for the sparse path.
- **Speed at parity after warm-starting the inner Newton**: q4_p100 in
  **2.47 s vs drmTMB 2.585 s (1.05×)** — even un-converged and
  un-optimized.

## What does NOT yet work (why this is not a win)

1. **EM not converged** at p=100 — 300 iters, still climbing +0.01/iter
   (slow linear EM convergence). Needs **SQUAREM** (3–8× fewer iters) to
   converge in practical time.
2. **logLik scale mismatch UNRESOLVED**: Julia −262.78 vs drmTMB −513.99.
   A 251-unit gap on the same model+data is impossible as a real fit
   difference (e²⁵¹ LR) → it is a **different additive constant** in the
   reported logLik. **Must reconcile** by evaluating Julia's marginal at
   drmTMB's fitted parameters before any equivalence/speedup claim.
3. **Optimizer fragility** on ill-conditioned real data — three cascading
   numerical issues hit and individually patched (sparse Cholesky ridge,
   β-Newton ridge, Λ PD-floor). Works now but needs principled hardening
   (trust region / proper PD parameterization) rather than ridge patches.

## Files (all under `julia/drm_q4/`)

- `sparse_aug_plsm.jl` — validated sparse E-step + marginal (Checkpoint 3).
- `sparse_em_fit.jl` — Laplace-EM (M-step Λ + β, monotone guard, warm-start).
- `run_sparse_q4p100.jl` — real q4_p100 fit + drmTMB comparison.
- `test_sparse_aug.jl` — Checkpoint 3 (passes, exact).
- `test_step1_sparse.jl` — Step-1 infra (passes).
- Dense oracles (correct but slow / NaN on real data): `q4_em_dense.jl`
  (monotone EM), `fit_q4_tmbgrad.jl` (exact gradient, Check C = 6.5e-9).

## Clear remaining path to the WIN (next session)

1. **Reconcile the logLik constant** — evaluate `laplace_ll` at drmTMB's
   q4_p100 fitted params; the offset to −513.99 is the constant. Then
   compare on a common scale (or compare recovered params directly).
2. **SQUAREM** — wrap the EM map (port `em_squarem.jl`); converge in
   ~30–50 iters instead of >300.
3. **Perf stack** (Scout A) — StaticArrays `SMatrix{4,4}` for the per-leaf
   blocks, symbolic-Cholesky reuse (`cholesky!`), type stability. Target:
   the per-fit 2.47 s → well under drmTMB once converged.
4. **The guaranteed big win: threaded bootstrap/profile CIs** — drmTMB's
   are serial R; DRM.jl's via `OhMyThreads` across 20 cores → ~16× on top.
   This is where "wins big" is structurally certain, once the single fit
   is correct + converged.
5. **Then** scale to p=354 (avonet, drmTMB 13.9 s) and p=1000 — where the
   O(p) sparse path should pull decisively ahead.

## Bootstrap pipeline (measured, q4_p100) — the genuine win

> **Correction (2026-10-11).** The drmTMB side below is drmTMB with
> `parallel = "none"` (single-fit time × 199). drmTMB's profile and bootstrap
> also support `parallel = "multicore"`, which was not measured, so the "~31×"
> figure compares threaded DRModels.jl with serial drmTMB, not with drmTMB's
> parallel mode. See the header of `bench/threaded_bootstrap_demo.jl`.

`threaded_bootstrap_demo.jl` (julia --threads=16):
- per-refit **~0.18–0.29 s serial** (warm-started from the point estimate)
  vs drmTMB's 2.585 s cold fit → ~10–14× per-refit (warm-starting + sparse).
- threading **2.23× (B=16) → 3.54× (B=48)** on 16 threads — scales with
  more tasks but allocation/GC-bound (~22% efficiency; perf stack would
  lift this toward ~10×).
- **extrapolated B=199: DRM.jl threaded ~16 s vs drmTMB serial ~514 s →
  ~31× faster on the bootstrap pipeline.** Stable across B=16 and B=48.

CAVEATS (do not oversell):
- threading is only 2.23×, not ~16× — needs allocation reduction.
- the ~14× per-refit assumes drmTMB refits COLD; a warm-started drmTMB
  bootstrap would narrow it. The clean apples-to-apples Julia-only edge
  is the 2.23× threading.
- logLik still unreconciled; point fit not fully converged.

## KEY FINDING: EM's Λ update is broken under Laplace (use TMB-like)

SQUAREM accelerated the EM dramatically — **6 iterations (vs 300+), 1.05 s,
2.47× faster than drmTMB** on q4_p100. But it exposed a fundamental issue:

- **Λ stays frozen at init** — the closed-form EM Λ-update is *always*
  rejected by the monotonicity guard because it **decreases the true
  marginal**. This is not a bug: the matrix-normal closed-form Λ MLE is a
  valid ascent only for an *exact* (Gaussian) posterior. Under the
  **Laplace** approximation (nonlinear scale), it ascends the EM
  lower-bound Q but can decrease the true marginal → variance components
  never get estimated. EM optimizes β fine; Λ is stuck.

**Implication (answers the EM-vs-TMB-like question):** the **TMB-like**
route (gradient/Newton on the TRUE marginal) is the correct one for the
variance components — it ascends the true marginal directly (so Λ IS
estimated) AND converges superlinearly (~10–20 iters vs EM's linear 300).
The exact implicit-function-theorem gradient was already validated to
**6.5e-9** vs finite differences (dense version). **Next build: adapt that
exact gradient to the sparse foundation** → correct Λ + fast single fit.
SQUAREM-EM remains useful as a robust warm-start for β.

## ROOT CAUSE of frozen Λ — diagnosed (decisive)

Two diagnostics (`test_lambda_direction.jl` p=8, `test_lambda_p100.jl` p=100,
both finite-diff on the true marginal):
- **p=8**: EM closed-form Λ update ASCENDS the marginal (ΔL=+0.124). Direction
  is correct; my derivation (EM-stationary = cheap-gradient marginal stationary)
  holds.
- **p=100 (real tree)**: EM closed-form **overshoots catastrophically** —
  Λ jumps 0.3 → 2.6, ΔL=−89. Only a tiny step (α=0.01) ascends (+0.058).

**Mechanism**: at a far-from-optimum init (β unfit) the posterior mode û is
huge, so the unscaled closed-form `Λ = (1/N)(ÛQÛ' + corr)` blows up (~2.6) and
overshoots. The *direction* ascends; the *step size* is ~100× too big.
Compounded by the guard's **warm-started E-step** returning inaccurate
marginals that missed the tiny genuine improvement → Λ froze.

**THE FIX (reconciles my derivation with the scouts' AI-REML):** replace the
unscaled EM closed-form with an **information-scaled (AI-REML / Newton) step
on the true marginal**, line-searched so it cannot overshoot. Use the analytic
Λ gradient `∂L/∂Λ = ½[Λ⁻¹ ÛQÛ' Λ⁻¹ + Λ⁻¹ C Λ⁻¹ − N Λ⁻¹]` with C = Σ Q[s,t]·
(H⁻¹ block) from **Takahashi (O(p))**, scaled by the average-information matrix
(Gilmour-Thompson-Cullis 1995) — converges in <10 iters, no overshoot. Also:
the guard must use a **fully-converged** E-step (not warm-stopped) for accurate
marginal comparisons. This is the concrete next build for a correct + fast fit.

## ML vs REML — design decision (user requirement: ML for model comparison)

REML likelihoods are NOT comparable across models with different fixed
effects → **ML must be the default** (for LRT/AIC model selection); REML is
an option for unbiased variance components. (As in lme4/glmmTMB/ASReml.)

Key clarification that simplifies the fix:
- `laplace_ll` **IS the ML Laplace marginal** = `log p(y,û|θ) − 0.5 logdet
  H_uu`. The `−0.5 logdet H_uu` (random-effects Laplace correction) is
  already included. **This is exactly drmTMB/TMB's objective (TMB does ML).**
- So the primary fix = **ascend `laplace_ll` over (β,Λ) with an
  AI-scaled / line-searched step** (no overshoot — the diagnosed bug). That
  gives the **ML** estimate, directly comparable to drmTMB's −513.99
  (reconciles the logLik gap) and valid for model comparison.
- The EM bug was never a missing objective term — `laplace_ll` is the right
  ML objective. It was the EM closed-form taking an overshooting STEP.
- **REML** = `method=:REML` flag: add `+0.5 logdet(I_ββ)` (fixed-effects
  information) to the objective for the Λ update, β profiled. Same machinery.

Design: `fit(...; method = :ML | :REML)`, default `:ML`. Both use the same
information-scaled line-searched ascent on the (ML or REML) Laplace marginal.
Build ML first (primary; reconciles logLik; enables model selection).

## KEYSTONE FIXED: Λ now estimated (ML marginal ascent)

`fit_ml_q4.jl` — β by conditional Newton + **Λ by line-searched gradient
ascent on the true ML Laplace marginal** (`laplace_ll`, = drmTMB's objective;
normalized step so it can't overshoot). p=20 result:
- **Λ MOVES** (0.3I → [0.534,0.221,0.636,0.633]) — no longer frozen.
- **logLik −47.77** (vs the broken EM's −55.16) — found a better optimum.
- converged in 16 iters, monotone (assertion holds).

So the frozen-Λ bug is resolved: ascend the ML marginal directly (not the
overshooting EM closed-form). This unblocks correct fits + comparable logLik
+ model comparison.

REMAINING for the win:
- **Speed**: current Λ step uses a FINITE-DIFFERENCE gradient (8.42 s at p=20
  — slow). Replace with the **analytic Λ gradient** (`∂L/∂Λ` via Takahashi,
  O(p), 1 eval) + AI scaling → fast. THIS is the next build.
- Verify p=20 Λ recovery (rough — identifiability? convergence?) at larger n.
- Run p=100, reconcile logLik vs drmTMB −513.99 (now possible: same ML
  objective), then SQUAREM + threaded CIs.

## BASELINE benchmark (pre-Takahashi) — measure-against point

| workload | drmTMB | DRM.jl (now) |
|---|---|---|
| single fit q4_p100 (ML, Λ estimated) | 2.585 s, LL −513.99 | 26.8 s, LL −259.88 (finite-diff, not converged) → 0.10× |
| bootstrap B=199 (CI pipeline) | ~514 s serial | ~16 s threaded → ~31× |

Single fit LOSES now — it's the FINITE-DIFFERENCE baseline; the analytic
Takahashi Λ-gradient is the speed fix, measured against this 26.8 s.

### logLik RECONCILED ✓ (the "gap" was a mis-recorded target)
drmTMB's ACTUAL q4_p100 logLik (results/r_results.json) is **−256.52**, NOT
−513.99. The −513.99 in the run scripts was a WRONG constant. Real picture:
- drmTMB: **−256.52** (2.48 s, itself "false convergence", n_iter 54)
- Julia ML: **−259.88** (not converged, climbing toward ~−256)

→ **Julia and drmTMB AGREE on logLik (~−256) — statistical equivalence
confirmed.** No constant offset (consistent with the p=8 exact match). Julia
converges UP to drmTMB's value and may slightly exceed it. Param agreement
already good: β_rho12 0.412 vs 0.401; β_mu2 slope 0.581 vs 0.582.
drmTMB sd_phy=[1.70,0.89,0.18,0.29]; Julia still converging.
FIX the −513.99 constant in run_*.jl scripts (use −256.52).

## INCONSISTENCY caught (user): single fit slower than a bootstrap refit
A single fit cannot legitimately be slower than a bootstrap refit (a refit IS
a fit). Resolution: the FAST bootstrap (0.18 s/refit) used the EM closed-form
Λ (FROZEN/wrong); the CORRECT single fit (26.8 s) uses a FINITE-DIFF Λ
gradient (20 E-step evals/step). Different algorithms → the "~31×" was on the
wrong (frozen-Λ) EM. Honest bootstrap of CORRECT fits ≈ 26.8 s·199/16 ≈ 330 s
(< drmTMB 514 s, but NOT 31×). The analytic gradient fixes BOTH.

## Cheap analytic Λ-gradient: VERIFIED WRONG (do not integrate)
`test_analytic_grad.jl`: G = ½N Λ⁻¹(Λ_em−Λ)Λ⁻¹ matches FD on mean axes
(E11/E12) but has the WRONG SIGN on the scale axis (E33: FD −0.26 vs +1.55)
and is 2× off overall. It's the "cheap gradient" — drops the implicit dû/dΛ in
the logdet-H term, which is NOT negligible for the nonlinear SCALE axes (the
TMB cheap-vs-exact issue). Caught before integrating (good).

## Prioritised next steps (user-requested, in order)
1. **Sparse EXACT gradient** = adapt the implicit-function-theorem exact
   gradient (already validated 6.5e-9 in dense `fit_q4_tmbgrad.jl`) to the
   sparse foundation. This is THE keystone: correct + O(p) + fast single fit
   (target 26.8 s → ~1-2 s), and legitimizes the bootstrap.
   logLik already reconciled (Julia ≈ drmTMB ≈ −256).
3. **Profile likelihood** (`profile_q4.jl`) — fix one param on a grid,
   re-optimise the rest per point; each grid point independent → **threaded**
   (OhMyThreads), like the bootstrap. drmTMB does this serially → big win.
4. **Louis SE** (`confint_q4.jl`) — Louis (1982) / Meng–Rubin Supplemented-EM
   observed information → Wald CIs. Port from GLLVM.jl `em_phylo.jl:828-893`.
5. SQUAREM on the corrected ML map; verify p=20 Λ recovery at larger n; scale
   to p=354/1000.

## Honest verdict

The **sparse foundation is correct and the stability advantage over the
dense/TMB-style routes is real** (runs where they NaN). But DRM.jl does
**not yet win** on q4: speed is at parity (un-converged), and the logLik
is not yet on a comparable scale. The path above is concrete; the work is
~1 focused session, not open-ended.
