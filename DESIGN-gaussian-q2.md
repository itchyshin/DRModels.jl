# Gaussian `animal|relmat(1 + x | g)` with two coefficient SD formulas

Model (rho fixed 0, unlabelled block): y_i = X_i b + s0_g f0_g + x_i s1_g f1_g + e_i,
f0, f1 iid N(0, C) with C RAW (as the Gaussian slope-only route and native R),
s_k = exp(Z_k a_k), e ~ N(0, sigma_i^2). Exact marginal:
y ~ N(X b, diag(sigma^2) + (w0 w0') o Sigma0[g,g] + (w1 w1') o Sigma1[g,g]),
Sigma_k = D_k C D_k, w0 = 1, w1 = x. No Laplace.

Implementation (src/gaussian_lss.jl, next to `_drm_gaussian_structured_sd_slope`):
- `_drm_gaussian_structured_q2_sd`: validation/refusals (two matching targets,
  coef(intercept)+coef(slope) once each, single structured term, fixed sigma,
  ML only, algorithm :auto, provider input A (animal) or K (relmat) only,
  rank-deficient SD design, constant-x with identical designs).
- `_fit_structured_gaussian_q2_sd`: dense n x n Cholesky, ForwardDiff, LBFGS,
  same convergence test as the single-component route. Params
  (beta, beta_sigma, a0, a1). Cost O(n^3)/eval, n <= 5000 (as existing dense route).
Router (src/gaussian_core.jl): q2 detected via `_animal_intercept_slope(rhs; kind)`
plus a matching sd target; admitted through new opt-ins
(`allow_animal_intercept_slope`, new `allow_relmat_intercept_slope`) so the old
refusal stands without sd targets. Predict paths widened to Gaussian.
Bridge label flag widened to Gaussian.
Not done: REML, missing responses, conditional-field (BLUP) export, rho (labelled
`animal(1 + x | p | g)`), sparse route, phylo/spatial.
Risks: touches `_split_ranef`/`_check_structured_re_lhs` (additive kwargs, defaults
unchanged); G-level Woodbury would be faster than dense n x n.
