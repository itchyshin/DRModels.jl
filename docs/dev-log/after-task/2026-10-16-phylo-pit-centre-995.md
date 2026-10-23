# After-task: Gaussian phylo(1 | g) PIT centre (#995, with #996)

Refs #995 (and #996). PR #1078, draft. This file covers both sessions on the PR:
the first (commits f739a0d…8563231) and the d3 resume (dbe2d45 onward).

## 1. Goal

Owner decision (Option A): centre the Gaussian `phylo(1 | g)` quantile residual on the
fixed-effect mean `Xβ̂`, not on `fit.means[:mu]` (which holds `Xβ̂ + S û`), and fold in
#996 (per-tip prior variance σ_b²·C_ii). The resume brief had two outstanding items:
- (a) measure the residuals against drmTMB's quantile residuals;
- (b) guard the O(p²) per-tip `Q_cond` solve.

## 2. Implemented

- **`src/location_only.jl`: `LocOnlyRanef <: AbstractDict`.**
  - It wraps the `group => BLUPs` Dict with `mu_fixed = Xβ̂` and the `LocOnlyProblem`.
  - It is attached on the sparse L-BFGS route (ML, REML, penalised) and on the EM route, which stored no BLUPs before.
  - `_loconly_row_tip_variance` gives C_ii per row, cached per fit. Since dbe2d45 it reads `diag(Q_cond⁻¹)` from `takahashi_diag` (exact, O(nnz(L)) = O(p) on a tree); the per-tip sparse solve was O(p²).
- **`src/quantile_residuals.jl`.**
  - For these fits, `_ranef_marginal_mix` returns the fixed-effect centre and σ_b²·C_ii per row.
  - Since dbe2d45, `_cdf_value` evaluates the Gaussian marginal N(x'β̂, σ_e² + σ_b² C_ii) in closed form rather than by the 32-node Gauss–Hermite mixture.
  - The structured-PIT warning stays for every other structured route.
- **`src/gaussian_ranef.jl` (9a86d58).** `ranef(fit)` unwraps `LocOnlyRanef` to its Dict, as it already unwraps the temporal NamedTuple. From f739a0d to dc5180e it returned the wrapper, so `ranef(fit) isa Dict` was false and `setindex!` threw.
- **Tests.**
  - `test/test_995_phylo_pit_centre.jl` (first session; extended in the resume):
    - the residual against the closed-form marginal on height-1 and height-3 trees, for sparse, REML and EM fits;
    - `ranef(fit) isa Dict` on all three;
    - the EM route's side effects: its BLUPs match the sparse route, `r2_constant_sigma` refuses it, and `_variance_boundary` now runs on it.
  - `test/test_995_tip_variance_linear.jl` (resume):
    - C_ii exact per row on two non-ultrametric trees, with shuffled rows, unequal replication and absent tips;
    - allocation scaling at 4096 vs 1024 tips;
    - the PIT on a caterpillar tree, where C_ii runs from 0.15 to 7.05 (atol 1e-10);
    - the C_ii cache.
  - `test_lane_j4_predict_numerics.jl` now expects no warning on the sparse fit and keeps it for the dense `algorithm = :gls` fallback.
  - `test_quantile_residuals_re_axis.jl`'s lone-phylo reference is now centred on `Xβ̂`.
- **Parity harness.** `tools/parity_phylo_quantile_residuals.R` and its Julia half write `docs/dev-log/evidence/parity-phylo-quantile-residuals.tsv`; the write-up is `docs/dev-log/evidence/2026-10-16-995-phylo-quantile-residual-parity.md`.
- **NEWS.** The bullet moved from `NEWS.md` to the fragment `news/995-phylo-pit-centre.md`: news/README.md says PRs do not edit `NEWS.md`. The fragment now also covers the O(p) cost, `ranef()`, the EM side effects and the twin note.

## 3a. Decisions and rejected alternatives

- **(b) is fixed with a cheaper exact path, not a size threshold.**
  - The selected inverse makes the first call O(p) (0.21 s at 262,144 tips), so a threshold would gate a cost that no longer exists. A warning or error would also have refused a correct answer on large trees.
  - Takahashi on `Q_cond` was chosen over a tree-depth walk (as in `phylo_tree_height`) because it reads exactly the matrix the likelihood uses. It is also the leaf-variance idiom already in `gaussian_sparse_lss.jl:25-26` and `gaussian_structured.jl:748`.
- **Closed form instead of quadrature (resume; outside the brief's two items).**
  - The PR said the residual "is now exactly" the marginal, but the 32-node mixture is exact only while σ_b√C_ii/σ_e ≲ 1.
  - Its residual error is 5.4e-5 at 2.5, 6.2e-3 at 4, 0.020 at 5 and 0.12 at 10. It was 0.031 on the caterpillar test and 0.035 on the parity `strong_signal` case.
- **The ordinary Gaussian `(1 | g)` route keeps the quadrature.** It is pre-existing #760 behaviour with its own tests; changing it would widen the PR. See §5.
- **Not stacked on #1075 / the J3 stack; coordinated instead.**
  - #1075 does not touch `src/location_only.jl`.
  - The J3 stack's edits in that file are #1082 (main 3537–3543) and #1091 (main 3155–3160 and 3459–3599).
  - The whole PR touches three ranges of the file (head numbering at f1b2382):
    - head 113–157, new code after main 112;
    - head 3444–3446, replacing main 3399, inside `_fit_structured_gaussian_em`;
    - head 3644–3645, replacing main 3597, inside `_fit_structured_gaussian_sparse_lbfgs`.
  - **The one textual conflict:** main 3597 is next to #1091's rewrite of main 3596. Resolve it by keeping both edits: `reml ? _withnll(fit, reml_full_nll, reml_nllgrad!) : …` together with `LocOnlyRanef(…)`.
  - J3's own note on #1073 already proposes that whichever lands second rebases and reruns the other's tests.
  - The resume itself changed only head 120–122 (comment) and 138–146 of this file, which no J3 PR reaches.
- **"Marginal", not "population-level", in cross-engine text.** drmTMB uses "population-level" for its fixed-effect-only `newdata` prediction (σ̂ alone), a different residual.

## 3b. Evidence

- **(b) and closed-form red proofs, on PR head 8563231, local Julia 1.12.7.**
  - A run of the committed `test/test_995_tip_variance_linear.jl` is red in its allocation testset: ratio 15.9 against the bound of 8, and 1.88 GB at 4096 tips against the bound of 82 MB. The run then stops, because a failing top-level testset throws.
  - A standalone run of the file's PIT testset on 8563231 fails with a max error of 0.031 (at atol 1e-4, and so also at the committed 1e-10).
  - Both are green at dbe2d45 on Julia 1.12.7, 1.10.4 and 1.13.0: ratio 4.0, about 1 kB per tip, PIT error 2.9e-15.
- **`ranef()` red proof.** On dc5180e, `ranef(fit) isa Dict` fails in all six fits of `test_995_phylo_pit_centre.jl`. It is green at 9a86d58.
- **(b) timings** (Julia 1.12.7, 4-core container):

  | tips | old: bytes / s | new: bytes / s |
  |---|---|---|
  | 4,096 | 1.9 GB / 0.33 s | 4.2 MB / 0.003 s |
  | 16,384 | 30 GB / 5.4–7.6 s | 16.8 MB / 0.01 s |
  | 262,144 | not run | 268 MB / 0.21 s |

- **(a) parity: DRModels.jl 9a86d58 (residual code as dbe2d45) vs drmTMB ca3ddefa5 (0.7.1).**
  - Headline max |q_julia − q_drmTMB|: 1.457 on the fixture and 3.929 on strong signal.
  - Under either engine's convention the engines agree to ≤ 3.5e-9; logLik agrees to ≤ 6e-8.
  - Both converged: Julia `true`; drmTMB code 0, `is_converged` TRUE, pdHess TRUE.
  - drmTMB's quantile residual is exactly its conditional `(y − fitted)/σ` (1e-15). Its residual docs describe a third, fixed-effect-only residual `(y − Xβ̂)/σ̂`, which differs from DRModels.jl by 1.814 on the fixture.
  - Every case has constant C_ii, because drmTMB requires ultrametric trees. Per-tip C_ii is covered by DRModels.jl's own tests only.

## 4. Verification

- **Local, Julia 1.12.7.**
  - `test_995_tip_variance_linear` 10/10 and `test_995_phylo_pit_centre` 20/20;
  - `test_quantile_residuals_re_axis` 24/24 and `test_lane_j4_predict_numerics` 122/122;
  - `test_temporal_residuals` 26/26, `test_temporal_homtoep` 180/180, `test_parity_temporal` 208/208.
- **Other Julia versions.** The #995 files are green on Julia 1.10.4 and 1.13.0. The conda-forge 1.10 build warns about a CHOLMOD library-version mismatch, which is an environment quirk.
- **Engine.** `bench/run_sparse_tmb_nd.jl` gives logLik −256.5064 (−256.51), converged.
- **Full suite and CI.** Results are in the check-log entry `docs/dev-log/check-log.d/2026-10-16-phylo-pit-centre-995.md`.
- **Independent review.** Four lenses (src correctness, tests, parity harness, Rose) were run on the diff, each followed by an adversarial verifier. Confirmed findings are fixed in 9a86d58 and the commits after it, or listed in §5.

## 5. Open

- **drmTMB-side owner decision (queued as #1101; drmTMB not edited).** For structured-mean fits, should `residuals(type = "quantile")` be:
  1. conditional (what its code does);
  2. fixed-effect only (what its residual docs say);
  3. marginal (what DRModels.jl returns)?

  Only option 3 gives parity, and the docs should match whichever is chosen. The earlier PR body cited drmTMB#1513, which is unrelated (`mi()` quadrature). #1101 needs mirroring on the drmTMB tracker; this session could not write there.
- **LogNormal `phylo(1 | g)` quantile residuals are badly mis-calibrated, pre-existing (#1102).** Reproduced on `main` f71b292 and on this branch: max error 9.89 against the log-scale marginal; sd(q) 3.02 against 0.90.
  - The structured LogNormal route fits Gaussian on log y and keeps the log-scale conditional mean in `fitted()`. The residual then takes the log twice and double-counts σ_b.
  - Not changed here; it needs its own issue. Fixes: `src/lognormal.jl` should store the response-scale median, and `_ranef_marginal_mix` needs a LogNormal + `LocOnlyRanef` closed-form branch.
- **Ordinary Gaussian `(1 | g)` quantile residuals** use the same 32-node mixture, with the same error at large σ_b/σ_e (#1103).
- **Quantile residuals in the far upper tail (#1103).** The PIT round trip Φ⁻¹(clamp(Φ(z))) shared by all continuous families loses precision beyond |z| ≈ 5 and caps |r| at 8.13. For the Gaussian closed form the residual could be returned as z directly.
- **The dense `algorithm = :gls` phylo route (#1104)** stores `means = Xβ̂` on the tip-correlation scale (C_ii = 1), so its PIT is already the marginal one. Its #995 warning is a false positive: on the j4 test data, `fitted − Xβ̂` = 0.0 and the residual matches the closed form to 8e-14. The warning was left unchanged, and the j4 test comment now says why.
- **#981:** spatial / relmat / animal structured effects still warn.
- **Doc drift (#1104).** `fitted()`'s docstring says it returns the population mean, but on these routes it is conditional (`Xβ̂ + û`). Noted, not changed. The `residuals()` docstring was updated in this PR (`src/gaussian_core.jl`, the "On the mean" sentence).

## 6. Rose audit

An independent Rose-lens pass (claim vs evidence, scope, drift, license) was run on the diff at 9a86d58, plus the working copy.

**Verdict: PASS-WITH-FIXES.** It reproduced:
- the GH error table, the caterpillar errors (2.9e-15 closed form, 0.031 quadrature) and the C_ii cost figures;
- the test counts;
- every evidence number against the TSV.

It found no drmTMB GPL source in the diff; the harness only calls drmTMB's API, and the quotes are short documentation excerpts.

Fixes it prompted, all in the commit after f1b2382:
- the `residuals()` docstring now describes the closed-form route and qualifies the 32-node rule;
- the J3 overlap is stated for the whole PR, including the #1091 adjacency (§3a);
- a hand-completed commit SHA in the evidence note was wrong and has been replaced with `git rev-parse` output;
- the check-log entry was added;
- worked example: N/A (see below);
- the NEWS O(p) bullet now states only the new cost, since `main` never computed C_ii;
- the j4 test comment now gives the correct reason for the `:gls` warning;
- the follow-ups are on the ledger (#1101–#1104).

## 7. Definition of Done

- **Implementation:** yes.
- **Tests:** yes; failing first where a new behaviour was added (§3b).
- **Docstrings:** no new public symbol, so none needed; `residuals()`'s existing docstring was updated.
- **Worked example:** N/A. No public API changed; `residuals(fit; type = :quantile)` is called as before.
- **Check-log:** `docs/dev-log/check-log.d/2026-10-16-phylo-pit-centre-995.md`.
- **After-task:** this file.
- **Rose audit:** §6.
