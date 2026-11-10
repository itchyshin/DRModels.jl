# Gaussian ordinary `(1 + x + z | g)` (q3, q4) with coefficient-specific SD formulas

## Model (confirmed from native R: `src/drmTMB.cpp` re_cov block, `R/drmTMB.R` q3/q4 gate, tests b29)

For group g with rows i in g, coefficient vector b_g in R^q (q = 3 or 4),
    b_g ~ N(0, Sigma_g),   Sigma_g = D(w_g) R D(w_g),
    D(w_g) = diag(exp(Z_k[g,] a_k), k = 1..q),   R a q x q correlation matrix.
    y_i = X_i beta + u_i' b_g + e_i,  e_i ~ N(0, exp(2 X^sigma_i beta_sigma)),
u_i = (1, x_i, z_i[, ...]) the member design row. R's TMB code builds
`VECSCALE(UNSTRUCTURED_CORR_t(theta), sd_g)`, i.e. exactly D R D with a full
estimated correlation matrix and a per-group SD from the sd() formula rows;
members without an sd() formula keep one constant SD. SD predictors must be
constant within group (R and Julia `_sd_group_design` both refuse otherwise).
Exact marginal (no Laplace): y_g ~ N(X_g beta, S_g + U_g Sigma_g U_g'),
S_g = diag(sigma_i^2). Unlabelled blocks only, ML only.

## Parameterisation
R: theta (q(q-1)/2) feed TMB `UNSTRUCTURED_CORR_t`: unit-diagonal lower-triangular
L with free strictly-lower entries, Sigma0 = L L', R = scale(Sigma0). I use the
same map, so theta is directly comparable to R's `theta_re_cov` (pair order
(2,1),(3,1),(3,2),(4,1)... row-major of the lower triangle). Julia theta block is `:recov`.
Vector layout: beta_mu | beta_sigma | a_1 | ... | a_q | theta(q(q-1)/2),
each `a_k` a block named by `_sd_target_block` (as the q2 structured route).

## Reuse
* Parsing/refusals/targets: `_sd_target_parts`, `_sd_target_block`, `_sd_group_design`
  (constant-in-group check), `_group_index`, rank check as q2.
* Fit scaffolding: LBFGS + ForwardDiff Hessian, `_vcov_from_hessian`, `_unitfree_converged`,
  `DrmFit`, `_withnll`, `_withiterations` exactly as `_fit_structured_gaussian_q2_sd`.
* New: `_re_multi_corr_vars` (q3/q4 lhs parser; `_re_kind` stays q1/q2 only),
  router leg in `gaussian_core.jl` before `_re_kind` in the sd() dispatch.
* There is no existing ordinary q3 Julia route (only `(1 + x | g)`), so the
  nesting check is internal: the all-`~ 1` fit vs an independent dense
  likelihood at its own optimum, and w-model logLik >= constant-model logLik.

## Likelihood evaluation
Per group, Woodbury with L_g = D_g Lc (Lc = Cholesky of R):
logdet V = logdet S + logdet(I + L' U' S^-1 U L),
V^-1 r = S^-1 r - S^-1 U L (I+M)^-1 L' U' S^-1 r. Cost O(n q^2 + G q^3) per
evaluation, no n x n matrix. R is built from theta each evaluation (q <= 4).
(Avoids the 5000-row cap of the dense q2 route.)

## Refusals kept
Same-group requirement, one block only, no other RE / sigma RE / meta_V /
structured terms, missing responses, REML, labelled block, q >= 5, duplicate or
unknown `coef()` selectors, rank-deficient SD design, non-mean targets,
`algorithm != :auto`. Existing `sd()` refusals for other shapes are unchanged.

## Not done
REML, partial q-block with other REs, labelled blocks (`(1+x+z | p | g)`),
conditional modes / BLUPs, predict/simulate for this fit, correlation block
exposed as tanh/rho (raw theta only), bridge/R-label wiring.
