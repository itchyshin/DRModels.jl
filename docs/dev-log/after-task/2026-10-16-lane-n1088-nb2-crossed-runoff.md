# After-task: lane n1088 — NB2 crossed run-off at heavy overdispersion (#1088)

## 1. Goal

#1088: `drm(bf(y ~ 1 + x + (1|g) + (1|h), sigma ~ 1), NegBinomial2())` at NB2 size ≈ 0.002
ended at θ = [-222, -24, 3021, 616, 687] with `converged = false`. Its own objective was 1034.05
there and 426.15 at drmTMB's optimum. The lane had to warn or stop on the run-off, reach drmTMB's
point, record one #1073 comparison row per seed, and add a real fit landing at θσ̂ in (3, 4)
next to the θσ = 3.5 point evaluation in `test_j3a_gradient_hessian_vcov.jl`. Stacked on #1075
(`740521f4`). Shannon, with Engine A (optimiser), Simulator (reproduction and DGP) and Rose
(claim vs evidence). One adversarial-review workflow ran: 4 reviewers (logic, test robustness,
claims vs evidence, merge and conventions) and 6 skeptic verifiers. No other subagents.

## 2. Implemented

- **Diagnosis (reproduced).** The data generator twin from the issue reproduces the three
  run-off objectives to the printed digits (1034.05 / 724.21 / 1145.06), the s7 / n = 1500 end
  point [-222.18, -24.37, 3021.5, 616.2, 686.8], and 426.146 at drmTMB's point. The n = 800 end
  point printed in the issue thread ([-162.4, 5.6, 705.2, 326.5, 350.7]) came from #1075's
  earlier head `0b72541f`. On `740521f4` it ends at log σ 1542.6 with the same objective.
  The run-off is one step. The MoM start caps the NB2
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
- **Always box-constrained:** rejected because it changes the path of every fit on five
  families' routes, for fits that never left the box. An earlier argument here, that
  collapsed log SDs would then end at about -7.9 and change the SE path, came from a run at
  `g_tol = 1e-6`. The adversarial review measured it at the route's `g_tol = 1e-8`, and there
  the box ends within 4e-4 of -8 on the #1088 sets (-8.0, -8.0, -7.99964) and within 2e-11 on
  `test_j3a`'s n = 300 collapse fit, inside the 1e-3 window that keeps the value path. So
  that argument does not hold and is withdrawn.
- **Making the gradient zero past a clamp:** rejected on its own. The run-off point would then
  look stationary and be reported as converged.
- **The lower side** (Poisson / no-dispersion limit, variance collapse) is a legitimate
  boundary and is not treated as a run-off (#422 / #959 own it). On the upper side the
  likelihood falls to -∞ whenever there is one non-degenerate observation (NB2 size → 0 drives
  log f(y) → -∞ for any y > 0; likewise Gamma, Beta, and BetaBinomial with 0 < y < n). On
  degenerate data (all-or-none BetaBinomial, all-zero Tweedie) the supremum itself lies past
  the clamp. The warning says so, and the fit is still marked not converged, because the
  reported value is the clamp, not an estimate.

## 3b. Evidence

Local runs: Julia 1.12.6 (conda-forge), `--check-bounds=yes`, BLAS 1 thread. drmTMB 0.7.1 was
installed with `R CMD INSTALL` from a source checkout at `ca3ddef` (GitHub), against R 4.4.3
and TMB 1.9.23 from conda-forge. The `drmTMB_commit` in each `ref.toml` is set by hand through
`DRMTMB_COMMIT`, not read from the installed package.

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
- Green on `5bc2fce` (37/37 and 45/45) and again on `3152d2f` after the review fixes.
- The re-fit costs 1.7–3.7 s per fit, and only on fits that ran off (first pass + SE alone:
  0.4–2.0 s). Measured with three other test processes sharing 4 cores, so these are upper bounds.
- q = 4 engine (`bench/run_sparse_tmb_nd.jl`): logLik −256.5064, 116 iterations, identical on
  base `740521f4` and on this branch. The route is not touched; wall time is not claimed.
- Regression reruns on the fix: 15 of 16 files pass (`crossed_laplace_generic` 53/53,
  `crossed_se_forwarding` 21/21, `betabinomial_crossed_laplace` 12/12, `finite_hessian_nan`
  38/38, `crossed_aghq_families`, `twin_gap_761`, `numerical_guards`, `boundary_polish`,
  `binomial_clamp`, `poisson_crossed_laplace`, `crossed_cumulative`, `twin_gap_569`,
  `twin_gap_739`, `ll_sanity_fuzzer`, `lane_j4_predict_numerics`). The four group-1 files ran
  with a call counter on `_crossed_nuisance_box_refit`, which counted 0 calls: on the route's
  existing tests the re-fit never fires and the optimiser path is the base one.
  `crossed_tweedie` (one fit takes over 36 CPU-min here, on base too) is still running at the
  time of this commit, with the same counter.

## 4. Open / follow-ups

- The same clamp-gradient inconsistency exists on `_fit_crossed_mean_laplace` (fixed-aux
  families) and on the phylo / general nuisance routes (`:898`). They are not touched here and
  have not been probed for run-off.
- The MoM start floor `max(s_MoM, 0.5)` is what makes the first step so large at size ≲ 0.01.
  An unfloored start reaches drmTMB's point on the three sets; on n = 800 that is the code-1
  point at 198.2952, above the box re-fit's 198.2334. It would change every NB2 crossed fit,
  so it is left for an estimates-affecting slice.
- The run-off recovery is demonstrated on NB2 only. On Gamma, Beta, BetaBinomial and Tweedie
  the guard is inert on the existing tests (0 re-fit calls), and no run-off case has been
  constructed for them.

## 5. Rose audit (claim vs evidence, scope, drift, license)

Verdict: **clean after the fixes below.**

- **Claims vs evidence.** Every number in the PR body, the commit messages and this report
  traces to a local run (the logs listed in 3b) or to a fixture. The adversarial review
  confirmed one unsupported claim (the always-box argument, measured at the wrong `g_tol`).
  Four more were overstated or stale: "reproduced all three end points", "the MLE cannot sit
  on the upper side" (false for degenerate BetaBinomial / Tweedie data), the provenance note,
  and a test comment. All were corrected in `3152d2f`. Speed: only the re-fit's extra cost is
  given, as an upper bound measured under contention. The q = 4 engine's wall time is not
  claimed.
- **Scope honesty.** The fix runs on five families' routes, but recovery is shown only for
  NB2 (section 4). The SE ladder is a verbatim port from J3e and is labelled as such. The
  n = 800 set is labelled not comparable (drmTMB code 1) and is not counted as a parity pass.
  The drmTMB s7 / n = 1500 SEs are all NaN, so only its optimum is compared.
- **Verification gap.** CI does not run on this stacked PR (the workflow triggers on PRs
  into `main` only). Every green and red here is local: Julia 1.12.6, one machine. The
  Julia 1.10 leg of CI has not been exercised for these tests.
- **Drift.** `report/` makes no claim about NB2 crossed convergence at heavy
  overdispersion. `docs/src/capabilities.md` lists NB2 crossed as Tested, which is still true.
  No doc change is needed. The NEWS line is in the PR body (lane rule; no NEWS.md edit).
- **License.** The R generator calls drmTMB and writes numbers. No drmTMB source is in the
  diff, and the drmTMB checkout lives only in the session scratchpad
  (`git ls-files | grep -i drmtmb/R/` is empty).
