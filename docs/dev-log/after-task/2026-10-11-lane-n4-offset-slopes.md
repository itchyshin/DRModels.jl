# After-task: lane N4 — Poisson offset + `(1|g)` (#941) and drmTMB-parity random-slope layouts (#960)

Speaking as Shannon. Perspectives active: Shannon (coordination), Hopper (engine), Rose
(license). No subagents ran.

## 1. Goal

Owner decisions of 11 Oct 2026: (7b) thread `offset()` into the Poisson random-effect
likelihoods so Poisson + offset + `(1 | g)` fits and matches drmTMB; (6b) port the
random-effect layouts drmTMB already fits, parity only — Gaussian correlated random-slope
mean + `sigma` random intercept first, then independent slopes `(1|g)+(0+x|g)` for Gamma
and Student. Not in scope: #943/#945 (sigma covariates, lanes J4/J5); correlated Gamma/Student
slopes (drmTMB refuses them).

## 2. Implemented

| Item | Where | Route |
|---|---|---|
| Poisson `offset()` + one `(1 \| g)` | `src/poisson.jl` (admission rule, `_fit_poisson_ranef`, `_fit_poisson_ranef_aghq`), `src/ordinary_laplace.jl` + `src/sparse_laplace_glmm.jl` (`_poisson_phylo_laplace_fg(…; offset)`) | default `:LA` (GHQ-32), `:AGHQ`, `:Laplace` |
| Gaussian `(1 + x \| g)` mean + `sigma ~ (1 \| g)` | new `src/gaussian_rs_sigma_ranef.jl`; dispatch in `src/gaussian_core.jl` | per-group 3-D Laplace, `marginal = :Laplace` |
| Gamma / Student `(1 \| g) + (0 + x \| g)` | new `src/indep_slopes_laplace.jl`; dispatch in `src/gamma.jl`, `src/student.jl` | TMB-convention Laplace, dense inner Newton + IFT lift (the #725 machinery) |

The offset is a constant added to η: it is independent of θ and b, so the Laplace
gradient, Hessian and log-determinant derivations are unchanged. Still refused with an explicit
error: offset with a structured marker, a random slope, crossed effects, `zi`/`hu`, `marginal = :VA`;
offset on a non-Poisson family under `:Laplace`.

## 3. Verification (independent runs; drmTMB 0.7.1 @ 0eb0467 built from source and only RUN)

Real data (the datasets in the issues, regenerated from the issues' own `prep.R`; not vendored):

| Model | DRModels.jl | drmTMB | 
|---|---|---|
| Owls Poisson `calls ~ food + offset(log BroodSize) + (1\|nest)`, `:Laplace` | −2603.20403 | −2603.20403 |
| Owls `food * parent` version, `:Laplace` | −2601.38800 | −2601.38800 |
| ChickWeight Gamma `(1\|chick)+(0+time\|chick)`, `sigma ~ diet` | −2306.17619 | −2306.17619 |
| ChickWeight Gaussian `(1+time\|chick)`, `sigma ~ time+diet+(1\|chick)` | −2313.48477 | −2313.48477 |
| ChickWeight Student indep. slopes, `sigma ~ 1` | −2437.88341 | −2437.88341 |

(The Owls default `:LA` GHQ-32 value −2603.1888 differs from the Laplace −2603.2040 by the
Laplace approximation error, as designed.) ChickWeight Student with `sigma ~ time + diet`:
drmTMB reports false convergence (logLik −2438.92, ν→ boundary); DRModels.jl reaches −2399.24 —
not used as an oracle. At the full 578 rows the Student route takes ~180 s (dense 100×100 joint
Hessian under ForwardDiff); the committed tests use 98-row simulated fixtures.

Committed fixtures are own simulated data (`test/fixtures/n4/`, generator and drmTMB output in
`gen_oracle.R` / `oracle_output.txt`), so no AGPL/GPL data enters the MIT tree.

## 3a. Decisions and rejected alternatives

- Optimiser: BFGS first (then bounded L-BFGS/BFGS restarts) for the two new routes. A centred L-BFGS
  stalled 4.8 logLik short on ChickWeight (near ρ = −1 ridge); BFGS reaches drmTMB's optimum in 5–25
  iterations. Interior starts, not the no-sigma-RE fit (it sits on the ρ = ±1 boundary).
- New files rather than edits inside the verified crossed-intercept and `_fit_correlated_ranef_gaussian`
  code, to leave verified paths untouched and keep conflicts with lanes J1–J5 minimal.
- `:AGHQ` is not offered for the independent-slope routes (Laplace only, as drmTMB).
- Not fixed (found, out of scope): `marginal = :AGHQ` on Poisson `(1|g)` throws "AGHQ Hessian at the
  mode must be negative" on the simulated rate fixture with or without an offset (pre-existing
  fixed 12-step Newton unroll). Worth its own issue.

## 4. Files touched

`src/poisson.jl`, `src/ordinary_laplace.jl`, `src/sparse_laplace_glmm.jl` (one keyword),
`src/gaussian_core.jl` (dispatch + docstring), `src/gamma.jl`, `src/student.jl`, `src/DRModels.jl`
(two includes), new `src/gaussian_rs_sigma_ranef.jl`, `src/indep_slopes_laplace.jl`;
`test/test_lane_n4_offset_slopes.jl`, `test/test_twin_gap_727.jl` (the refusal it asserted now fits),
`test/fixtures/n4/*`; `docs/src/diagnostics-and-validation/implementation-map.md`;
`news/n4-offset-re-and-parity-slopes.md`; `docs/dev-log/check-log.d/2026-10-11-lane-n4-offset-slopes.md`.
No q=4 file edited, so the verified q=4 baseline (logLik −256.51, 2.18×) is untouched.

## 5. Tests

Julia 1.10.4 (conda-forge), `--check-bounds=yes`, BLAS=1, one process per test file (the repo
runner is fail-fast, so the full-suite run was split per file; 4 files excluded by the runner's
own `_TEST_EXCLUDE` list, gated parity/JET suites not run).

- New `test_lane_n4_offset_slopes.jl`: **42/42** pass (oracles: drmTMB 0.7.1 on own fixtures).
- Touched suites: `test_twin_gap_727.jl` **15/15** (refusal case rewritten), `test_twin_gap_745.jl` **28/28**.
- All 369 discovered files: **367 pass, 2 fail**, ≈15,338 passing assertions (summed from per-file summaries), 2 failing, 0 errors.
- The 2 failures are pre-existing and identical on a pristine `origin/main` worktree:
  `test_bootstrap_formula_structured.jl` (Poisson phylo bootstrap CI: `-8.925 < -9.120 < -1.048`, same digits on main) and
  `test_q4_reml_vcov.jl` (relerr `0.001036048912923572 < 0.001`, same digits on main). Both look Julia-1.10 / numeric-environment
  specific (the repo targets 1.12/1.13); neither path is touched here.
- Dry-run merges (`git merge-tree`) of this branch against the five open J-lane branches: all clean.

## 6. Rose (license)

No drmTMB source was copied. drmTMB was built from its public repository and run to produce
oracle numbers; only generated outputs are recorded. Fixtures are own simulated data.
