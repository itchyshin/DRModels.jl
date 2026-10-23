# `src/experimental/` — unwired variants, with verdicts

**Nothing in this directory is loaded by `module DRModels`.** Every file here is either a
recorded negative result, a superseded predecessor of a production route, or a
diagnostic oracle kept for reproducibility. The module docstring (`src/DRModels.jl`) is the
authoritative wiring list; this README records *why each file is here*, so the
directory reads as a set of findings rather than unfinished work. Completion-roadmap
Wave B pass, 2026-08-27 (D-179 arc).

**The rule:** do not wire anything from here into the module without a GitHub issue
making the case — several files below are kept precisely because they demonstrate an
approach that measurably fails, and wiring one by accident would resurrect a defect
the project already paid to diagnose.

## Recorded negative results (kept as evidence, not aspiration)

| file | verdict |
|---|---|
| `fit_em_natgrad.jl` | **#13 decision-gate FAIL (2026-08-01):** natural-gradient EM stalls vs the sparse TMB-style route on `q4_p100`. `algorithm = :natgrad` is deliberately not exposed. The reusable piece — the Fisher metric on log-Cholesky parameters — was extracted to `src/lc_metric.jl`, which *is* wired. |
| `fit_em_closed.jl` | Built on the closed-form Λ M-step, which **descends** the true Laplace marginal at p=100 for every step size tested — the #472 measured finding, characterised by `test/test_lambda_p100.jl` (wired, asserts the descent). |
| `em_squarem_fit.jl` | SQUAREM acceleration of the same Laplace-EM family. Acceleration of a route whose Λ step does not ascend (#472) is moot; parked with its host. |

## Superseded by the production engine (`src/fit_q4_sparse_tmb.jl`)

| file | what it was |
|---|---|
| `fit_q4_tmbgrad.jl` | The TMB-style exact outer-gradient proof of concept — the idea that *became* the production engine. Historical; the production file carries the verified O(p) implementation. |
| `fit_ml_q4.jl` | ML by line-searched gradient ascent on the marginal — the fix for the overshooting closed form, before the exact-gradient engine landed. |
| `fit_ml_warm.jl` | Warm-started E-step variant of the finite-difference Λ ascent. Correct direction, superseded on speed. |
| `fit_q4_p100_tmb.jl` | A clean `q4_p100` driver for the tmbgrad POC (worked around a monitoring bug in its harness). Driver only. |
| `estep_armijoguard.jl`, `estep_initprior.jl`, `estep_lm.jl`, `estep_trustregion.jl` | Four competing hardenings of the inner mode-finder against the cold-start divergence at p ≥ 500. The production mode-finder solved this in `sparse_aug_plsm.jl`/`fit_q4_sparse_tmb.jl`; these are the exploration record. |

## Diagnostic oracles (kept for reproducibility)

| file | role |
|---|---|
| `q4_em_dense.jl` | Dense-Σ⁻¹ correctness oracle and moderate-p benchmark for the q=4 model. |
| `fit_sparse_direct.jl` | Direct LBFGS on the validated sparse marginal — the instrument that separated "EM stalls" from "phylo identifiability" at p=20. |

## Leftover prototype

| file | status |
|---|---|
| `location_only.jl` | The conjugate location-only solver was **promoted**: the wired version lives at `src/location_only.jl` (`algorithm = :em`, #12). This copy is the pre-promotion prototype and is retained only as history; edit the wired file, never this one. |
