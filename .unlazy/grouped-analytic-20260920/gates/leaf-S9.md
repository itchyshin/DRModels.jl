# Gates: leaf-S9 DRModels q4 route, measure-first on the two named sections (overnight arc 2026-09-20)

OWNS: bench/profile_q4_sections.jl, bench/results/q4_sections_*.tsv

DO NOT TOUCH: src/ (this slice is measurement only unless GC.2 opens a change), test/test_q4_perf_identities.jl, PR #781 (ready for review, not ours to disturb).

BASELINE: bench/results/q4_sections_5000691b4.tsv. p = 5,000: fit wall 43.576 s, fd_vcov 243.117 s, 220 chol factorisations, 0 reuse fallbacks. p = 1,000: fit 9.074 s. p = 100: fit 0.992 s. The handover names two unmeasured shares at p = 5,000: the beta-block trace at about 23% and the Gst/v assembly at about 17%.

- [x] GC.1: `bench/profile_q4_sections.jl` splits the p = 5,000 fit wall further into the beta-block trace, the Gst/v assembly, and the remainder, plus numeric factorisation count against numeric factorisation cost. Sections sum to within 10% of the measured fit wall.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --project=. bench/profile_q4_sections.jl --gate sections_fine --p 1000,5000
  EXPECT: GATE GC.1 PASS and a TSV at bench/results/q4_sections_<sha>.tsv carrying the new columns
  EVIDENCE: GATE GC.1 PASS (sha 8780340f8, TSV bench/results/q4_sections_8780340f8.tsv, no other Julia process running during the timed reps). New `--gate sections_fine` installs measurement-only, VERBATIM top-level redefinitions of `DRModels.marginal_and_exact_grad`/`DRModels.laplace_ll` (re-read from this worktree's HEAD, `chol_ref` threading included) with `time_ns()` brackets, calling through UNMODIFIED to `estep_mode`/`_estep_fast`/`_estep_robust`/`sparse_pd_chol` (S5's cholesky!-reuse untouched) and to `joint_nll`/`joint_nll_T`/`joint_grad_T`/`takahashi_selinv`/`leaf_hess`/`leaf_hess_du`. First attempt installed the redefinition lazily inside a helper function (guarded, called once) — `methods()` showed it correctly replacing the original, but a literal probe print in the new body never fired across a full fit (0 hits): the optimizer's already-JIT'd call graph did not pick up a mid-run redefinition. Fixed by moving the redefinition to plain top-level statements (this file's own leaf-S3 precedent) — confirmed by a probe print firing on every call, and by `--gate loglik --p 100` still passing (measured -256.5273 vs target -256.51, diff 0.0173) after the redefinition is active, i.e. it is behaviourally identical to the original, not a shadow with different numerics.

  Sections as %% of the chosen-rep fit wall (reps=3, chol_reuse_fallbacks=0 both p):

  | section | p=1000 (fit_wall 8.254s) | p=5000 (fit_wall 39.681s) |
  |---|---|---|
  | beta_trace | 1.571s (19.032%) | 9.626s (24.257%) |
  | gst | 0.141s (1.704%) | 0.825s (2.078%) |
  | v_assembly | 1.294s (15.677%) | 7.227s (18.213%) |
  | gst+v_assembly | 1.435s (17.381%) | 8.052s (20.291%) |
  | estep_mode_wall | 4.796s (58.101%) | 19.484s (49.102%) |
  | joint_grad_T | 0.249s (3.013%) | 1.525s (3.843%) |
  | logdetP_chol | 0.065s (0.785%) | 0.322s (0.812%) |
  | takahashi | 0.038s (0.459%) | 0.167s (0.420%) |
  | joint_nll_T | 0.029s (0.357%) | 0.155s (0.390%) |
  | kron_prior | 0.032s (0.387%) | 0.165s (0.415%) |
  | other (residual) | 0.040s (0.485%) | 0.187s (0.470%) |

  `chol_factorizations` (existing counter, unaffected by this gate) = 289 (p=1000), 220 (p=5000); `chol_reuse_fallbacks` = 0 both. `estep_mode_wall` is the WHOLE inner Newton mode-search call, not isolated CHOLMOD-factorisation cost — see GC.2/item-3 finding below; the already-tracked factorisation COUNT is reported alongside it, not conflated with it.

  Handover's asserted shares (at p=5000): beta-block trace ~23% — HOLDS (measured 24.257%, +1.3pp). Gst/v assembly ~17% — HOLDS closely at p=1000 (measured combined 17.381%) but is measurably higher at p=5000 (measured combined 20.291%, +3.3pp / ~19% relative); "Gst" alone is only 2.078% at p=5000, so the ~17% figure only holds under the combined (Gst+v_assembly) reading, not a Gst-only reading.

- [x] GC.2: THE DECISION. If one section exceeds 30% of the fit wall AND an identity-safe change is obvious from the measurement, make exactly ONE change and gate it as an identity at rtol 1e-8 against origin/main. Otherwise STOP: the measurement is the deliverable and the next lever is named by number for the next arc.
  CHECK: read the TSV, state each section's share, and write the decision with its numbers
  EXPECT: a written decision either way
  EVIDENCE: DECISION: STOP — no src/ change made. `estep_mode_wall` is the only section over 30% (58.101% at p=1000, 49.102% at p=5000), but no identity-safe change is obvious from this measurement: `estep_mode_wall` is a black-box timing of the WHOLE `estep_mode` call (fast-Newton or robust-LM path, however many `sparse_pd_chol` factorisations plus however many `joint_nll`/`joint_grad` line-search evaluations that path takes) — this gate never redefines `_estep_fast`/`_estep_robust`/`sparse_pd_chol` themselves (that would shadow S5's cholesky!-reuse pattern cache, the exact functions S5 changed), so it cannot say how much of that 49-58% is CHOLMOD factorisation cost vs. gradient/line-search cost. `beta_trace` (24.257%) and `gst`+`v_assembly` (20.291%) are both under 30% individually and combined (44.5%) are two DIFFERENT non-mergeable code spans, so neither licenses a single one-line change either. Per the gate's own "when in doubt, STOP" clause, no src/ edit is made.

  NEXT LEVER (named, for the next arc): add `time_ns()` brackets directly around each `sparse_pd_chol(...)` call site inside `_estep_fast` (src/sparse_aug_plsm.jl, the `n_newton` loop, ~lines 445-475) and `_estep_robust` (src/sparse_aug_plsm.jl, ~lines 477-520), accumulated into a counter parallel to the existing `CHOL_FACTORIZATIONS`/`CHOL_REUSE_FALLBACKS` globals, so `estep_mode_wall` can be split into "CHOLMOD wall" vs. "Newton gradient/line-search wall" before any change to the Newton loop can be justified as measured rather than guessed. This is a src/ change (a new counter/timer, not a behaviour change) and is out of scope for this measurement-only slice.

- [ ] GC.3: only if GC.2 opened a change. Fitted parameters and logLik equal origin/main within rtol 1e-8 at p = 100 and p = 1,000, and the q4 identity suite still passes.
  CHECK: env JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 julia --check-bounds=yes --project=. test/test_q4_perf_identities.jl --gate nll && ... --gate logdet && ... --gate vcov
  EXPECT: each GATE ... PASS
  EVIDENCE: N/A — GC.2 decided STOP; no src/ change was made, so this gate does not apply. `test/test_q4_perf_identities.jl` was not touched or run.

## STOP conditions

Any src/ change without GC.2 written first. Any edit to test/test_q4_perf_identities.jl. Any action on PR #781. An identity failing at rtol 1e-8. A Julia suite started while the GLLVM lane is running one.
