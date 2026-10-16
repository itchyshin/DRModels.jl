# After-task: lane n1088 — NB2 crossed run-off at heavy overdispersion (#1088)

## 1. Goal

#1088: `drm(bf(y ~ 1 + x + (1|g) + (1|h), sigma ~ 1), NegBinomial2())` at NB2 size ≈ 0.002
ended at θ = [-222, -24, 3021, 616, 687] with `converged = false`. Its own objective was 1034.05
there and 426.15 at drmTMB's optimum. The lane had to warn or stop on the run-off, reach drmTMB's
point, record one #1073 comparison row per seed, and add a real fit landing at θσ̂ in (3, 4)
next to the θσ = 3.5 point evaluation in `test_j3a_gradient_hessian_vcov.jl`. Stacked on #1075
(`740521f4`). Shannon, with Engine A (optimiser), Simulator (reproduction and DGP) and Rose
(claim vs evidence). No subagents.

## 2. Implemented

- **Diagnosis (reproduced exactly).** The data generator twin from the issue reproduces all
  three end points and objectives to the printed digits: 1034.05 / 724.21 / 1145.06 at the
  run-off, and 426.146 at drmTMB's point. The run-off is one step. The MoM start caps the NB2
  size at 0.5 (log σ = 0.35) while the MLE is ≈ 0.0025 (log σ ≈ 3), so the start gradient is
  ≈ [227, 26, -3021, -617, -688]. LBFGS's first, unscaled step (α = 1, accepted by Armijo because
  1034.05 < 2979.84) lands past every clamp. There the objective is flat but the analytic
  gradient is not: `_laplace_v123_nuisance(:nb2_fixed)` returns `-2r·dl/dr` at the clamped `r`,
  and `gσ` is evaluated at the clamped log SD. So the next line search fails.
- **Fix** (`_fit_crossed_mean_laplace_nuisance`, all five families on the route). If the first
  pass ends past an UPPER clamp (`_nuisance_runoff`: log-dispersion > `_dispersion_clamp(kind)`,
  a log SD > 3, or a non-finite estimate), the route re-fits from the same start with
  `Fminbox(LBFGS)` inside the clamp box (`_crossed_nuisance_box_refit`: log-dispersion
  [-b, b], log SDs [-8, 3]). The re-fit is adopted only if it strictly lowers the objective. A
  fit that still ends past an upper clamp warns (`_warn_nuisance_runoff`; silent in
  bootstrap / profile refits) and is marked `converged = false`. Fits that stay inside the box
  take exactly the path they took before. `recover_runoff = false` (internal kwarg, also on
  `_fit_nb2_crossed_laplace`) switches the re-fit off, for the warning test.
- **SE tolerance ladder, ported from J3e** (`_SE_MODE_TOLS`, `_se_grad`, verbatim from
  `a32f4d4`), on this route only. The requested real fit at θσ̂ in (3, 4) showed that the
  cold 1e-12 SE inner solve stalls on its rounding floor at a stencil point (heavy-tailed
  counts, y up to 5.7e5). The gradient-path Hessian then returns an all-NaN vcov. The same
  point converges at 1e-10. The text is identical to J3e's, so merging J3e only needs one
  copy kept.

## 3a. Decisions and rejected alternatives

- **Box on the upper side only:** rejected. The collapsed log SDs then run off downwards (to
  ≈ -1e9), and on s8 the search missed the interior SD (nll 386.13 vs 384.07).
- **Step-capped LBFGS (`InitialStatic(scaled = true)`) as the re-fit:** it reaches drmTMB's
  point on all three sets, but nothing stops it leaving the box again. On n = 800 it stops at
  drmTMB's code-1 point (198.2952), while the box re-fit finds 198.2334.
- **Always box-constrained:** rejected. It changes every fit on five families' routes, and
  collapsed log SDs would end at about -7.9 instead of at or past -8, which changes the SE path
  (#1073's tests depend on it).
- **Making the gradient zero past a clamp:** rejected on its own. The run-off point would then
  look stationary and be reported as converged.
- **The lower side** (Poisson / no-dispersion limit, variance collapse) is a legitimate
  boundary and is not treated as a run-off (#422 / #959 own it). The upper side cannot hold an
  MLE: NB2 size → 0 drives log f(y) → -∞ for any y > 0.

## 3b. Evidence

Local runs (Julia 1.12.6, `--check-bounds=yes`, BLAS 1 thread; drmTMB 0.7.1 `ca3ddef`,
R 4.4.3, TMB 1.9.23, all installed from conda-forge in this container).

| set | drmTMB `opt$convergence` | `sdr$pdHess` | log σ drmTMB | log σ DRModels | nll drmTMB | nll DRModels | status |
|---|---|---|---|---|---|---|---|
| nb2_runoff_s7_n1500 | 0 | true | 3.0036 | 3.0036 | 426.145749 | 426.145749 | optimum compared (sdreport SEs all NaN) |
| nb2_runoff_s8_n1500 | 0 | true | 2.9663 | 2.9663 | 384.074142 | 384.074142 | optimum compared; SEs of β, log σ and log SD h within 0.92% (value path) |
| nb2_runoff_s7_n800 | 1 | true | 3.1402 | 3.03963 | 198.295187 | 198.233426 | not comparable (code 1); DRModels lower |
| nb2_disp3to4_s1_n5000 | 0 | true | 3.20298 | 3.20298 | 848.089764 | 848.089764 | gradient path, no warning, SEs within 1% |

- Red on base `740521f4` (tests-only commit `1779615`): `test_1088_nb2_crossed_runoff.jl`
  6 pass / 18 fail / 1 error (the helpers do not exist there). Base ends at log σ 3021.5 /
  2902.0 / 1542.6, with nll 1034.05 / 1145.06 / 724.21. `test_j3a_gradient_hessian_vcov.jl`
  has 36 pass / 9 fail; the (3, 4) fit ends at log σ 2573.5, nll 920.92.
- With the run-off fix but without the ported ladder, `test_j3a` has 42 pass / 3 fail: the
  (3, 4) fit reaches drmTMB's point, but its SEs are NaN.
- Green on `5bc2fce`: 37/37 and 45/45.
- The re-fit costs 1.7–3.7 s per fit, and only on fits that ran off (first pass + SE alone:
  0.4–2.0 s). Measured with three other test processes sharing 4 cores, so these are upper bounds.
- q = 4 engine (`bench/run_sparse_tmb_nd.jl`): logLik −256.5064, 116 iterations, identical on
  base `740521f4` and on this branch. The route is not touched; wall time is not claimed.
- Regression reruns of the touched crossed-route test files: in progress at the time of
  this commit; recorded in the PR body when they finish.

## 4. Open / follow-ups

- The same clamp-gradient inconsistency exists on `_fit_crossed_mean_laplace` (fixed-aux
  families) and on the phylo / general nuisance routes (`:898`). They are not touched here and
  have not been probed for run-off.
- The MoM start floor `max(s_MoM, 0.5)` is what makes the first step so large at size ≲ 0.01.
  An unfloored start reaches the optimum on these sets, but it changes every NB2 crossed fit,
  so it is left for an estimates-affecting slice.

## 5. Rose audit

Pending: done once the regression reruns are in.
