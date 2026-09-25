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
  with more than one random field instead of drawing only one. For a phylo
  field it now maps rows to tree leaves by name (`_phylo_mean_leaf_index`, as
  the fits do), not by first-seen order: before, a data set whose species order
  differed from the tree's drew the field on the wrong tips (sister-tip
  covariance -0.005 against 0.294 in the model, 6000 draws), and a tree with
  tips absent from the data fell back to the conditional `simulate`, which on
  the `meta_V` route has no random field. This was also wrong on main for
  plain `phylo(1 | sp)`, which the same fix repairs. A `meta_V` + random-effect
  fit whose simulator cannot be built (e.g. no `tree`) now refuses instead of
  falling back. Found by the Arc 2 verifier pass.
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
Bootstrap simulator: on tip-ordered data the draws are bit-identical before and
after the leaf-mapping fix for plain `phylo`, `phylo` with `sigma ~ x`,
`meta_V + phylo`, `meta_V + (1 | study)`, `(1 | study)` and Poisson `phylo`
(checked by a one-off old-vs-new run); the eight bootstrap test files are green.

## Found, not fixed (outside this domain)

- `y ~ x + relmat(1 | id) + (1 | g)` WITHOUT `meta_V` silently drops `(1 | g)`
  (same df and logLik as `relmat(1 | id)` alone, measured); the dense
  structured route has no ordinary-bar term. drmTMB's bridge already refuses
  this shape (`drm_julia_refuse_structured_with_ordinary_bar`).
- The marginal bootstrap simulator draws only the FIRST structured field for
  the two-structured route (`phylo + relmat`) as well.
- The dense Gaussian phylo fallback route (e.g. `phylo(1 | sp)` with
  `sigma ~ x`) maps rows to tips by first-seen order in the FIT: reordering the
  rows of one data set changes its logLik (-25.402 in tip order, -28.489 with
  species first seen as L1, L3, L5, ...; measured). The sparse default route is
  row-order invariant. Not fixed here (a different route; fixing it changes
  that route's answers).
- `test_bootstrap_formula_structured.jl` Poisson multi-height round-trip fails
  on base da8b3f871 too (identical numbers).

## Not covered

REML, random slopes, `spatial`, `sigma` random effect, `sd(g) ~ ...`, a
dense V, missing responses, two fields on one grouping column; bootstrap
with more than one field. No interval-coverage claim.
