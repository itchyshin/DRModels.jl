# After-task: rebase #1080 (J3c) onto current main via #1079

## 1. Goal

Finish PR #1080 after #1079. Rebase onto the updated J3b head (current `main`
+ #1099 + J3b) so `pull_request` CI can run. Keep the PR a draft. Shannon;
Noether on the SE contract. No subagents.

## 2. Implemented

- Cherry-picked the five unique J3c commits onto #1079, then rebased onto
  `edec98493` (the docs-only CI retrigger on `a2df49c12`):
  `e6095f16a`, `92cefb586`, `90c12190b`, `c8e7631eb`, `67faafc2d`. All applied
  without conflict.
- J3c SE sites still called the one-shot `_se_grad(θ -> …)`. Updated
  `_fit_phylo_mean_laplace_hetero`, `_fit_phylo_mean_laplace` and
  `_fit_cumulative_phylo_laplace` to pass `(θ, tol)` into the shared `#1099`
  ladder (same follow-up J3b needed).
- `_dispersion_clamp` now lists `nb2_hetero` / `gamma_hetero` / `beta_hetero`
  at the same ±4 bound as their `_fixed` twins. The J3a review made unknown
  kinds throw; the old #1080 branch had used a `::Val = 4.0` catch-all, which
  is gone. No catch-all restored.
- Hetero clamp route (`eta=` on `_se_hessian_path`) kept.

PR body: **Refs #1073**. No `Closes` — J3c is one lane of #1073.

## 3. Evidence

Local: Julia 1.10.12, `--check-bounds=yes`, BLAS 1 thread.

| case | result |
|---|---|
| `test/test_j3_se_reference.jl` | 112/112 (70 J3b + 42 J3c) |
| `test/test_j3c_hetero_clamp_hessian.jl` | 12/12 |
| `test/test_j3a_gradient_hessian_vcov.jl` | 45/45 |
| `test/test_1088_nb2_crossed_runoff.jl` | 62/62 |
| full `julia --project=test --check-bounds=yes test/runtests.jl` (375 files) | **16,817 passed / 0 failed** (741 testsets), EXIT 0, 05:19:51–14:26:31Z |

License: fixtures are generated drmTMB numbers only.

## 4. Open / follow-ups

- #1073 stays open (J3d / remaining routes).
- #1079 CI is running on `a2df49c12` in parallel.

## 5. Rose audit

Verdict: **clean for the rebase claim.** Unique work is still the three J3c
routes plus the hetero per-observation clamp. The ladder adaptation and
explicit hetero `_dispersion_clamp` methods are rebase fixes required by
#1099 / J3a, not a new SE-accuracy claim. Status stays draft. No `Closes`.

## 6. CI retrigger

The force-push landed while the PR still targeted `#1079`'s branch. Retargeted
to `main`; this prose commit is a `synchronize` so CI can start. The full
local suite on this tree is unchanged (16,817 / 0).

## 7. Julia 1 shard 1/4 (run 39136437802)

`test/test_j3c_hetero_clamp_hessian.jl:66` failed on Julia 1.13.2:
`fit.theta[end] = -8.119 ≯ _SE_LOGSD_LO = -7`. Reproduced locally. The
engine path is fine: the 1.13 `Distributions.NegativeBinomial` redraw is
not the 1.10 draw, and that 1.13 `y` collapses phylo log-SD to the fg
floor. The unit `_se_hessian_path` assertions still passed (11/12).

Fix: freeze the 1.10.12 `y` in the e2e block so the fallback stays the
ησ-clamp case on both CI versions. No `src/` change. Still **Refs #1073**,
no `Closes`.

Local after the pin:

| case | result |
|---|---|
| `test_j3c_hetero_clamp_hessian.jl` Julia 1.10.12 | 13/13 |
| `test_j3c_hetero_clamp_hessian.jl` Julia 1.13.2 | 13/13 |
| focused J3 + #1088 (1.10.12, `--check-bounds=yes`) | 232/232 |
