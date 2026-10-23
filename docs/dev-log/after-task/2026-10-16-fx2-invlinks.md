# After-task: FX2 inverse links (sweep-oct16 t6)

## 1. Goal

Fix the three unambiguous `_param_response` inverse-link bugs found on lane t6
(pins drmTMB `ca3ddefa5`, DRModels.jl `f71b292`). Shannon, with Noether
(engine contract) and Rose (claim vs evidence). No subagents.

## 2. Implemented

- `src/gaussian_core.jl` `_param_response` / `_link_deriv` / `_mean_response`:
  - Student `:nu` → `2 + exp(η)` (was `exp(η)`). Derivative stays `exp(η)`.
  - SkewNormal `:nu` → identity (was `exp(η)`). Derivative is 1.
  - ZeroOneBeta `:mu` → logistic (was identity). Derivative is `μ(1-μ)`.
- Docstrings: `predict` lists ZeroOneBeta with the logit families;
  `ZeroOneBeta` now says `predict` is the interior beta mean while `fitted`
  stays the unconditional mixture.
- Test: `test/test_fx2_invlinks.jl` (auto-discovered). Red before the patch
  (11 failures; Student in-sample `≈` hid a difference of 2 on the
  near-Gaussian ridge — the kernel `_param_response(Student(), :nu, [0.0])`
  was the honest red). Green after.

## 3a. Decisions and Rejected Alternatives

- **LogNormal μ is a methods choice — not changed.** Julia stores and predicts
  `exp(η)` (response-scale median; `lognormal.jl` docstring). drmTMB
  `type = "response"` returns identity / log-Y μ (documented, #1320). Options
  are reported in the FX2 receipt, not chosen here.
- Unknown `:nu` families now error instead of silently `exp`. Only Student,
  SkewNormal, and Tweedie carry `:nu` on this pin.

## 4. Verification

See the PR body for the red-proof transcript, the green re-run, and the full
suite result. C17 not recertified (no drmTMB `R/` / `src/` change).

GitHub issue write returned 403; this PR Refs #1092 and the t6 draft
`ISSUE_predict_invlinks.md` on `sweep-oct16/t6`. Never Closes/Fixes.
