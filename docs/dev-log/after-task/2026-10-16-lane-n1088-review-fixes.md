# After-task: #1099 review fixes — refit counter, 1e-8 SE warning, Gamma run-off

## 1. Goal

Reviewer notes on PR #1099 (head `e827da37`, stacked on #1075). Shannon; no subagents.
(1) The Tweedie crossed test can pass by construction because inner fits use `se = false`;
assert the box-re-fit counter is nonzero. (2a) Warn or record when the SE gradient falls
back to the 1e-8 inner tolerance. (2b) Add one Gamma or Tweedie run-off test
(`recover_runoff = true` is the default). (3) Keep the tolerance-ladder `_se_grad` as the
single definition; note that whichever of #1079 / #1099 lands second deletes the duplicate.
Never `Closes`; `Refs #1088`.

## 2. Implemented

- `_CROSSED_NUISANCE_BOX_REFITS`: incremented by `_crossed_nuisance_box_refit`. The three
  NB2 recovering fits and the new Gamma case assert it is `> 0`.
- `_SE_MODE_TOL_USED` + `_warn_se_mode_fallback`: `_se_grad` records the loosest inner
  tolerance a Hessian stencil actually used; the crossed-nuisance SE block warns once when
  that is `1e-8`. Silent inside bootstrap / profile refits. 1e-10 is recorded but quiet
  (the (3, 4) NB2 fit in `test_j3a` stays warning-free).
- One Gamma run-off (`test/fixtures/j3_se/gamma_runoff_s7_n200/data.csv`; generator in
  `test/parity/gen_1088_nb2_runoff_data.jl`). `_fit_gamma_crossed_laplace` forwards
  `recover_runoff` (default `true`). Tweedie was not chosen: inner `se = false` and 36+
  CPU-min per public fit.
- `_se_grad` remains the single ladder definition in this tree. #1079 still has a one-shot
  `fgtight(θ)` twin; the PR body records that the second of #1079 / #1099 to land deletes
  the duplicate.

## 3. Evidence

Local: Julia 1.10.10, `--check-bounds=yes`, BLAS 1 thread.

| case | base `740521f4` | head |
|---|---|---|
| `_SE_MODE_TOL_USED` / `_warn_se_mode_fallback` | `UndefVarError` | 10/10 pass |
| Gamma public `drm()` on `gamma_runoff_s7_n200` | `converged = false`, log SDs 18.55 / 25.32, nll sentinel 1e18 (2 fails) | 12/12 pass; box-re-fit count 1 |
| `test_1088` (NB2 + new sets) | NB2 loop errors on missing counter (helpers absent) | 27 + 12 + 10 + 12 pass |
| `test_j3a` | n/a this pass | 9+7+3+26 pass; (3, 4) fit still warning-free |

Gamma recovered fit (direct, `se = false`): θσ = 1.277, log SDs 2.03 / 2.13, nll = −4466.
Public `drm()` (se = true) recovers the same point; the gradient-path Hessian at a stencil
point is non-finite, so SEs are NaN with the existing failed-stencil warning. Point recovery
is the claim; Gamma SEs at this DGP are not claimed.

## 4. Open / follow-ups

- #1079's `_se_grad(fgtight(θ))` still exists on that branch. Second-to-land deletes it.
- Beta / BetaBinomial still have no constructed run-off case.
- Gamma public SEs on `gamma_runoff_s7_n200` are NaN (non-finite stencil gradient). Not
  filed: the helper already reports NaN rather than a made-up SE, and there is no drmTMB
  Gamma row for this DGP.

## 5. Rose audit

Verdict: **clean for the review-fix claims.** Recovery is now shown for NB2 and Gamma, not
Tweedie (cost / `se = false`). The 1e-8 warning is unit-tested; the (3, 4) NB2 fit is not
claimed to hit that rung. License: fixture numbers only. CI does not trigger on this stacked
PR (workflow `pull_request` branches: `[main]` only); leave draft until a green run exists.
