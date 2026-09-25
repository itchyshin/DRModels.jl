# After-task: Arc 2, `meta_V` with a random effect on the mean (2026-09-24)

## Scope

One Arc 2 true-parity domain: make DRModels fit drmTMB's model for Gaussian
meta-analysis with known sampling variances plus a random intercept on the
mean, so drmTMB's bridge can lift its `meta_v_with_random_effect` refusal.
Branch `claude/arc2-metav` from da8b3f871.

## What changed

- `src/gaussian_meta.jl`: new `_fit_meta_gaussian_re`, the exact marginal
  `N(Xb, diag(v + sigma^2) + sum_k s_k^2 Z_k C_k Z_k')` by whitened Woodbury;
  `meta_V` docstring names the new shapes.
- `src/gaussian_core.jl`: `drm(..., Gaussian())` dispatches `meta_V` plus any
  `(1 | g)` / `phylo` / `relmat` / `animal` intercepts to it, before the
  structured and meta-only routes that each dropped half the formula. Refuses
  slopes, `spatial`, a shared grouping column, `:sparse`, and a penalty.
- `src/inference.jl`: the marginal bootstrap simulator refuses a `meta_V` fit
  with more than one random field instead of drawing only one.
- Tests: `test/test_meta_random_effect.jl` (registered in `runtests.jl`).
- Docs: `NEWS.md`, `docs/src/capabilities.md`, receipt under
  `docs/dev-log/evidence/arc2-metav-random-effect/`.

## Evidence

Seven fits on six fixtures against drmTMB 709efbeb0 `engine = "tmb"`: equal
df, max abs dlogLik 2.1e-10, max relative estimate difference 2.0e-9. Through
drmTMB's R bridge (label builder patched in-session) df, logLik and fixed
effects also match; see the receipt for two R-side reporting gaps.

## Neighbours (D-273)

`meta_V` alone and with `sigma ~ x` still dispatch to `_fit_meta_gaussian`
(byte-identical coefficients and logLik, tested); plain `(1 | g)` still takes
`_fit_ranef_gaussian` (tested); existing router test files all green.

## Found, not fixed (outside this domain)

- `y ~ x + relmat(1 | id) + (1 | g)` WITHOUT `meta_V` silently drops `(1 | g)`
  (same df and logLik as `relmat(1 | id)` alone, measured); the dense
  structured route has no ordinary-bar term. drmTMB's bridge already refuses
  this shape (`drm_julia_refuse_structured_with_ordinary_bar`).
- The marginal bootstrap simulator draws only the FIRST structured field for
  the two-structured route (`phylo + relmat`) as well.
- `test_bootstrap_formula_structured.jl` Poisson multi-height round-trip fails
  on base da8b3f871 too (identical numbers).

## Not covered

REML, random slopes, `spatial`, `sigma` random effect, `sd(g) ~ ...`, a
dense V, missing responses, two fields on one grouping column; bootstrap
with more than one field. No interval-coverage claim.
