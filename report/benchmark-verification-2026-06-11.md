# Benchmark verification — q4 PLSM single fit + O(p) scaling

**Date:** 2026-06-11 (run 2026-06-10 evening MDT)
**Branch:** `shannon/bench-verify` (worktree `DRM-bench`, off `5bb3b6c`)
**Machine:** Apple M1 Ultra, 20 cores; Julia 1.10.0; BLAS threads = 1.
**Scripts:** `bench/run_sparse_tmb_nd.jl` (single fit), `bench/run_scaling.jl` (scaling).
**Author perspective:** Shannon (no subagents running).

Purpose: independently re-run DRM.jl's two headline performance benchmarks and
compare the freshly measured numbers against the documented claims in
`HANDOVER.md` §2 and `report/comparison-grid.md`. This is a verification pass, not
new science. The repo discipline ("verify before claiming; do not promote
extrapolated numbers to measured") is applied throughout: every number in the
**Measured** columns below came out of a run on this machine in this session.

---

## 1. Single-fit q4 PLSM (head-to-head vs drmTMB) — REPRODUCED

`bench/run_sparse_tmb_nd.jl`, real `q4_p100` fixture (p = 100 species, 1 obs each),
sparse TMB-like LBFGS, Λ0 off the diagonal singularity. Two independent
invocations (each does a warm-up fit + 2 timed fits, reports best-of-2).

| Quantity | Documented (claim) | Measured (this session) | Verdict |
|---|---|---|---|
| logLik (Julia) | −256.51 | **−256.5177** (deterministic, both runs) | ✅ matches |
| converged | true | **true** (iters 97, g_resid 7.2e-3) | ✅ *(#946, 2026-10-16: that `true` was Optim's f-plateau flag; with g_resid 7.2e-3 > g_tol 1e-3 the gradient-verified flag would read false. See `report/comparison-grid.md` §2 note.)* |
| wall (Julia) | 1.14 s | **1.098 s / 1.123 s** (two runs) | ✅ at/under claim |
| drmTMB baseline | 2.48 s | 2.48 s (documented constant; **not re-run** — see §4) | — |
| speedup | 2.18× | **2.21× / 2.26×** (= 2.48 / measured) | ✅ meets-or-exceeds |
| \|Δ logLik\| vs drmTMB | 0.01 | **0.0023** | ✅ |

Fitted fixed effects (this run): β_mu1 = [0.141, 0.389], β_mu2 = [0.123, 0.582],
β_s1 = [0.201], β_s2 = [−0.529], β_rho = [0.397];
sd_phy = [0.969, 0.510, 0.117, 0.172] (drmTMB reference [1.70, 0.89, 0.18, 0.29]).
The scale-axis sd_phy values (0.117, 0.172) sit at the Watanabe-singular variance
boundary, exactly as documented; both engines plateau there.

**Conclusion:** the 2.18× single-fit claim reproduces. The measured Julia wall
time (1.098–1.123 s) is at or slightly below the documented 1.14 s, so the
measured speedup against the documented drmTMB baseline is **2.21×–2.26×** — i.e.
the headline 2.18× is met and marginally exceeded on this hardware. The logLik is
deterministic at −256.5177 across runs.

---

## 2. O(p) scaling — modest sweep (p ∈ {100, 500, 1000}) — REPRODUCED, near-linear

`bench/run_scaling.jl` with `DRM_QGATE_PS=100,500,1000 DRM_QGATE_NREP=4`,
balanced + caterpillar trees, O(p) sparse-precision sampler. Deliberately stops at
p = 1000 (a few-minute run); p = 10,000 was **not** re-run in this pass (the
committed `report/qgate-multishape-scaling.md` already holds a measured full-range
run — see §3).

| shape | p | nobs | wall (s) | iters | logLik | logLik/nobs | converged |
|---|---|---|---|---|---|---|---|
| balanced | 100 | 400 | 0.723 | 32 | −765.77 | −1.914 | true |
| balanced | 500 | 2000 | 1.907 | 13 | −4776.47 | −2.388 | true |
| balanced | 1000 | 4000 | 7.432 | 21 | −8661.62 | −2.165 | true |
| caterpillar | 100 | 400 | 0.986 | 51 | −1082.22 | −2.706 | true |
| caterpillar | 500 | 2000 | 3.348 | 25 | −4606.06 | −2.303 | true |
| caterpillar | 1000 | 4000 | 10.174 | 41 | −9857.10 | −2.464 | true |

**Fitted scaling exponent** (k in wall ≈ p^k, log-log OLS over the three points;
independently recomputed in a separate Julia process to cross-check the script):

| shape | k (this run) | endpoint k (t₁₀₀₀/t₁₀₀) | documented k |
|---|---|---|---|
| balanced | **0.94** | 1.01 | 1.08 (full range to 10k) |
| caterpillar | **0.97** | 1.01 | — |

All six fits converged with finite wall time and finite logLik; the gate's own
verdict was **PASS**. Both exponents are ≈ 1 (sub-linear OLS slope is an artifact
of the p = 500 point landing below the trend line — it took only 13 iters vs
21/32 at the neighbouring sizes; the clean endpoint ratio 10× p → ~10.3× time
gives k ≈ 1.01). This is consistent with the O(p) claim: no super-linear blow-up,
flat-ish iteration counts, stable per-observation logLik.

**Conclusion:** O(p) scaling reproduces as near-linear (k ≈ 0.94–1.01) over
p = 100–1000. This is a *modest-range* confirmation; it does not by itself re-prove
the p = 10,000 figure (that lives in §3 as a separately-measured artifact).

---

## 3. Full-range scaling already on record (not re-run here, cited as measured)

The committed `report/qgate-multishape-scaling.md` (left untouched by this pass)
records a measured run over p ∈ {100, 1000, **10000**}, nrep = 4, on this same
engine:

| shape | p = 100 | p = 1000 | p = 10000 | k |
|---|---|---|---|---|
| balanced | 0.761 s | 7.528 s | **47.812 s** | 0.90 |
| caterpillar | 1.027 s | 9.338 s | **68.263 s** | 0.91 |

This corroborates the "near-perfect O(p) to 10,000 species" claim with a measured
p = 10,000 point (47.8 s balanced; the HANDOVER's 112.9 s figure is from an
earlier/different sampler configuration but the same order of magnitude and the
same near-linear k). I did **not** regenerate this in the present session — I am
citing the existing committed measurement and verified my modest sweep is
consistent with its low-p regime.

---

## 4. drmTMB (R) side — NOT independently re-measured (by design)

drmTMB *is* installed on this machine, and `bench/R/fit_r.R` contains a runnable
q4_p100 cell. I deliberately did **not** run it, for two reasons:

1. **License boundary** (CLAUDE.md / AGENTS.md): the R q4 path loads drmTMB via
   `devtools::load_all("…/drmTMB")` — i.e. it pulls the GPL drmTMB source into
   process. The repo rule is "R-parity uses generated outputs only; never vendor
   drmTMB GPL source." The documented drmTMB numbers (logLik −256.52, wall 2.48 s,
   "false convergence 8") are the sanctioned generated-output baseline, and the
   benchmark script bakes them in as constants.
2. **Scope:** the task is to verify the *DRM.jl* side with fresh runs. The Julia
   wall time is what changed; the drmTMB baseline is a fixed reference.

So the **2.21×–2.26× speedup is measured-numerator / documented-denominator**: a
fresh DRM.jl wall time over the recorded drmTMB wall time. The drmTMB side was not
re-timed in this pass. A clean-room head-to-head (R timing regenerated under the
license-safe fixture path, no in-process GPL load) remains the documented open
item (`comparison-grid.md` §7 NEEDS-REVIEW #2).

---

## 5. Third claim — "valid CIs where drmTMB's Hessian fails" — NOT re-run this pass

`comparison-grid.md` §6 documents this as: at the q4_p100 ML optimum the observed
information (17×17) has 1 negative eigenvalue (the Λ[4,3] singular direction) and
16 positive, yielding **16/17 finite Wald SEs** where drmTMB's `sdreport` is
all-NaN, plus a 60/60 successful parametric bootstrap. Reproducing this requires
the heavier `infer_q4` inference pipeline on the q4 fit, which is outside the two
benchmark scripts targeted here. I did **not** re-measure it in this session; I
report it as documented-but-not-re-verified-here. The wired diagnostic surface
(`check_drm`, `confint`) supports the mechanism (it explicitly handles the
non-PD-vcov / boundary case), but the specific 16/17 count was not regenerated.

---

## 6. Reproduced vs extrapolated vs not-re-run (summary)

**Reproduced this session (measured on this machine):**
- Single q4 PLSM fit: logLik −256.5177, converged, wall 1.098–1.123 s →
  **2.21×–2.26×** over the documented drmTMB 2.48 s baseline. (Claim: −256.51,
  2.18×. ✅ met/exceeded.)
- O(p) scaling p = 100→1000, balanced + caterpillar: all converged/finite,
  **k ≈ 0.94–1.01** (≈ linear). (Claim: O(p), documented k ≈ 1.08. ✅ consistent.)

**Measured but on record only (cited, not re-run this pass):**
- Full-range scaling to p = 10,000 (balanced 47.8 s, k = 0.90) —
  `report/qgate-multishape-scaling.md`.

**NOT re-measured this session (reported as documented, flagged honestly):**
- drmTMB R-side timing/logLik (license boundary; documented constant used).
- "16/17 valid Wald SEs vs drmTMB all-NaN" CI claim (needs the inference pipeline).

**Still extrapolated, NOT promoted (per discipline):**
- The "~12× vs drmTMB at p = 10,000" remains an extrapolation of drmTMB's measured
  slope, not a measured head-to-head. Not asserted as measured here.

---

## 7. Discrepancies / notes

- **No material discrepancy** on the two re-run claims. The single-fit speedup
  came in slightly *better* than documented (2.21–2.26× vs 2.18×), attributable to
  the M1 Ultra and a marginally faster Julia wall (1.10 s vs 1.14 s). The logLik
  is bit-identical across runs (−256.5177).
- The modest-sweep OLS exponents (0.94 / 0.97) sit just under 1.0 because of the
  p = 500 iteration-count dip; this is run-to-run optimiser noise, not a scaling
  regression — the endpoint ratio and the committed full-range k (0.90–0.91) all
  agree on "near-linear".
- `bench/run_scaling.jl` **overwrites** `report/qgate-multishape-scaling.md` on
  every run. My modest sweep transiently replaced the committed full-range file; I
  restored it via `git checkout` so the measured p = 10,000 artifact is preserved.
  (Worth noting for whoever runs the gate next — pass a different `OUT` or expect
  the clobber.)

---

*Verification by Shannon, 2026-06-11. Numbers in §1–§2 reproduced by independent
runs on Apple M1 Ultra / Julia 1.10.0. drmTMB side and the CI count were not
re-measured this pass and are flagged as such.*
