# After-task: Gaussian phylo(1 | g) PIT centre (#995, with #996)

Refs #995 (and #996). PR #1078, draft. This file covers both sessions on the PR:
the first (commits f739a0d…8563231) and the d3 resume (dbe2d45 onward).

## 1. Goal

Owner decision (Option A): centre the Gaussian `phylo(1 | g)` quantile residual on the
fixed-effect mean `Xβ̂`, not on `fit.means[:mu]` (which holds `Xβ̂ + S û`), and fold in
#996 (per-tip prior variance σ_b²·C_ii).

The resume brief had two outstanding items:
- (a) Measure the residuals against drmTMB's quantile residuals.
- (b) Guard the O(p²) per-tip `Q_cond` solve.

## 2. Implemented

- `src/location_only.jl`: `LocOnlyRanef <: AbstractDict` wraps the existing
  `group => BLUPs` Dict with `mu_fixed = Xβ̂` and the `LocOnlyProblem`.
  - It is attached on the sparse L-BFGS route (ML, REML, penalised) and on the EM route,
    which stored no BLUPs before.
  - `fit.means`/`fit.scales` were not used, because `sigma()` and the bridge enumerate them.
  - `_loconly_row_tip_variance` gives C_ii per row, cached per fit. Since dbe2d45 it reads
    `diag(Q_cond⁻¹)` from `takahashi_diag` (exact, O(nnz(L)) = O(p) on a tree). The
    per-tip sparse solve it replaces was O(p²).
- `src/quantile_residuals.jl`: for these fits, `_ranef_marginal_mix` returns the
  fixed-effect centre and σ_b²·C_ii per row.
  - Since dbe2d45, `_cdf_value` evaluates the Gaussian marginal N(x'β̂, σ_e² + σ_b² C_ii)
    in closed form; before, it used the 32-node Gauss–Hermite mixture.
  - The structured-PIT warning stays for every other structured route.
- Tests:
  - `test/test_995_phylo_pit_centre.jl` (first session) pins the residual against the
    closed-form marginal on height-1 and height-3 trees, for sparse, REML and EM fits.
  - `test/test_995_tip_variance_linear.jl` (resume) has three testsets:
    - C_ii exact per row on two non-ultrametric trees, with shuffled rows, unequal
      replication and absent tips;
    - allocation scaling at 4096 vs 1024 tips;
    - the PIT on a caterpillar tree, where C_ii runs from 0.15 to 7.05 (atol 1e-10).
  - `test_lane_j4_predict_numerics.jl` now expects no warning on the sparse fit and keeps
    it for the dense `algorithm = :gls` fallback.
  - `test_quantile_residuals_re_axis.jl`'s lone-phylo reference is now centred on `Xβ̂`.
- Parity harness: `tools/parity_phylo_quantile_residuals.R` and its Julia half write
  `docs/dev-log/evidence/parity-phylo-quantile-residuals.tsv`. The write-up is
  `docs/dev-log/evidence/2026-10-16-995-phylo-quantile-residual-parity.md`.
- NEWS: the bullet moved from `NEWS.md` to the fragment `news/995-phylo-pit-centre.md`
  (news/README.md: PRs do not edit `NEWS.md`). A second bullet was added for the O(p) cost.

## 3a. Decisions and rejected alternatives

- **(b) is fixed with a cheaper exact path, not a size threshold.** The selected inverse
  makes the first call O(p) (0.21 s at 262,144 tips), so a threshold would gate a cost that
  no longer exists. A warning or error would also have refused a correct answer on large
  trees.
  - I chose Takahashi on `Q_cond` over a tree-depth walk (as in `phylo_tree_height`)
    because it reads exactly the matrix the likelihood uses.
  - It is also the leaf-variance idiom already in `gaussian_sparse_lss.jl:25-26` and
    `gaussian_structured.jl:748`.
- **Closed form instead of quadrature (new in the resume, outside the brief's two items).**
  The PR said the residual "is now exactly" the marginal, but the 32-node mixture is exact
  only while σ_b√C_ii/σ_e ≲ 1. The residual error is 5.4e-5 at a ratio of 2.5, 6.2e-3 at
  4, 0.020 at 5 and 0.12 at 10. On the caterpillar test it was 0.031, and on the parity
  `strong_signal` case 0.035. For a Gaussian response the marginal is closed form, so the
  quadrature was dropped on this route only.
- **The ordinary Gaussian `(1 | g)` route keeps the quadrature.** It has the same error at
  large σ_b/σ_e, but it is pre-existing #760 behaviour with its own tests, and changing it
  would widen this PR.
- **Not stacked on #1075 / the J3 stack.** #1075 does not touch `src/location_only.jl`.
  The J3 stack's edits in that file are #1082 (main 3537–3543) and #1091 (main
  3155–3599). The resume touched only head lines 136–146 there, which none of them reach.

## 3b. Evidence

- **(b) red/green.** The forcing test file was run on PR head 8563231, locally on Julia
  1.12.7:
  - allocation ratio 15.9 (bound < 8), 1.88 GB at 4096 tips (bound 82 MB);
  - the PIT testset fails with a max error of 0.031.

  The same file is green at dbe2d45 on Julia 1.12.7 and 1.10.4: ratio 4.0, about 1 kB
  per tip, PIT error 2.9e-15.
- **(b) timings** (Julia 1.12.7, 4-core container):

  | tips | old: bytes / s | new: bytes / s |
  |---|---|---|
  | 4,096 | 1.9 GB / 0.33 s | 4.2 MB / 0.003 s |
  | 16,384 | 30 GB / 5.4–7.6 s | 16.8 MB / 0.01 s |
  | 262,144 | not run | 268 MB / 0.21 s |

- **(a) parity** (DRModels.jl dbe2d45 vs drmTMB ca3ddefa5 = 0.7.1):
  - The headline max |q_julia − q_drmTMB| is 1.457 (fixture) and 3.929 (strong signal).
  - Under either engine's convention the engines agree to ≤ 3.5e-9, and logLik to ≤ 6e-8.
  - Both engines converged: Julia `true`; drmTMB code 0, `is_converged` TRUE, pdHess TRUE.
  - drmTMB's quantile residual is exactly its conditional `(y − fitted)/σ` (1e-15).
  - The difference is by design: DRModels.jl returns the population-level PIT, drmTMB the
    conditional residual. Details and caveats are in the evidence file.

## 4. Verification

- Local (Julia 1.12.7):
  - `test_995_tip_variance_linear` 8/8, `test_995_phylo_pit_centre` 8/8,
    `test_quantile_residuals_re_axis` 24/24, `test_lane_j4_predict_numerics` 122/122;
  - `test_temporal_residuals` 26/26, `test_temporal_homtoep` 180/180,
    `test_parity_temporal` 208/208.
- Julia 1.10.4: both #995 files green. The conda-forge 1.10 build warns about a CHOLMOD
  library-version mismatch, which is an environment quirk.
- The full local suite and CI results are recorded in the check-log entry.

## 5. Open

- **drmTMB-side owner decision (queued, drmTMB not edited).** For structured-mean fits,
  should `residuals(type = "quantile")` be population-level or conditional?
  - Its docs say "fixed-effect, population-level"; its code conditions on the modes.
  - The earlier PR body cited drmTMB#1513, which is unrelated (`mi()` quadrature).
  - The decision still needs an issue on the drmTMB tracker; this session could not write
    there.
- Ordinary Gaussian `(1 | g)` quantile residuals use the same 32-node mixture, with the
  same error at large σ_b/σ_e. This is a follow-up and was not changed here.
- The dense `algorithm = :gls` phylo route stores `means = Xβ̂` on the tip-correlation
  scale (C_ii = 1), so its PIT is already the population-level one. Its #995 warning
  looks like a false positive (scout reading, not run). The warning was left unchanged.
- #981 (spatial / relmat / animal structured effects) still warns.
- `fitted()`'s docstring says it returns the population mean, but on these routes it is
  conditional (`Xβ̂ + û`). This is doc drift, noted and not changed.

## 6. Rose audit

See the PR body ("Rose audit"): an independent claim-vs-evidence pass was run on the
final diff, and its verdict and any fixes are recorded there.
