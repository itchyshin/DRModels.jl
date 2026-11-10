# Design: Gaussian sigma-side random effects with an SD formula

Branch `claude/joint-mu-sigma-sd-20261107`. Twins of two native drmTMB tests on
`origin/main`: `test-gaussian-mu-sigma-joint-sd.R` (6a) and
`test-gaussian-sigma-structured-sd.R` (6b). Status: both implemented (see
"Verdict"); this document records the model, the approximation, and what is
and is not covered.

## Model

Common form. Latent vectors are standardised and the SD formula only scales
them:

    y_i ~ N(mu_i + c_i * va[ga_i],  exp(2 eta_i))
    eta_i = (X_sigma beta_sigma)_i + d_i * vb[gb_i]
    c_i = exp(Za alpha_a)[ga_i]                  (6a only; mean intercept)
    d_i = loading_i * exp(Zb alpha_b)[gb_i]      (sigma intercept or slope)
    va ~ N(0, I),   vb ~ N(0, K)

* **6a (joint, ordinary).** `y ~ x + (1 | id)`, `sigma ~ 1 + (1 | site)` or
  `(0 + z | site)`, `sd(id) ~ w`, `sd(site, sigma[, coef(z)]) ~ v`.
  `K = I`. The groups may be crossed, so the 2x2 per-group block of the
  existing same-group route (`_fit_musigma_ranef_gaussian`) does not apply.
  `mu` and `sigma` effects are independent: no cross-parameter correlation,
  no extra parameter (R: independent `dnorm` priors on both latent vectors).
* **6b (structured sigma).** `y ~ x`, `sigma ~ 1 + animal(1|id)` or
  `relmat(1|id)`, `sd(id, animal|relmat, sigma) ~ z`. `vb ~ N(0, K)` with raw
  (non-normalised) `A` / `K`, `d_i = exp(Z alpha)[g]`. Mean is fixed. R refuses
  phylo here; spatial is a kernel with an estimated range, not ported.
  Plain `sigma ~ 1 + animal(1|id)` with no SD formula stays refused (known
  gap #963); only the SD-formula form is added.
* Parameter vector `theta = [beta_mu; beta_sigma; alpha_a; alpha_b]`.
  Blocks: `:mu`, `:sigma`, then one SD block per target (`:sd` when there is
  one target, the hashed `sdtarget_*` names of `_sd_target_block` when there
  are two, so `predict_parameters` resolves them unchanged).

## Likelihood approximation

Not Gaussian once sigma has a random effect. Laplace in the standardised
latent `v = (va, vb)`, which is exactly TMB's objective (the approximation is
invariant to the affine map `u = scale * v`, so it equals the raw-scale dense
Laplace that the R oracle uses):

    nll(theta) = F(v_hat) + 1/2 log det H(v_hat) + 1/2 log det K + n/2 log 2pi
    F = sum_i [eta_i + 1/2 r_i^2 e^{-2 eta_i}] + 1/2 va'va + 1/2 vb' K^-1 vb
    r_i = y_i - mu_i - c_i va[ga_i]

Closed-form gradient and Hessian in `v` (per observation, `ie = e^{-2 eta}`):
`dF/dA = -r ie`, `dF/deta = 1 - r^2 ie`, `d2F/dA2 = ie`, `d2F/deta2 = 2 r^2 ie`,
`d2F/dA deta = 2 r ie`. The joint density is **not convex** in `(A, eta)`
(the 2x2 block has determinant `-2 r^2 ie^2 < 0`), so the inner solver is a
damped Newton with Hessian regularisation off the basin and full Newton steps
inside it.

Outer gradient: forward-mode AD of the Laplace value with `v_hat` replaced by
one Newton step from the converged Float64 mode. At a stationary inner point
that step carries exactly `dv_hat/dtheta = -H^-1 dg/dtheta`, so the first
derivative is exact with no nested AD. Outer optimisation, boundary polish and
the scale-free convergence rule are the shared `_ordinary_laplace_optimize`.
The Wald covariance is central differences of that analytic gradient (the
value-difference Hessian in the shared helper returned a non-finite stencil
here).

## Reuse points

* `_sd_group_design` (constant-within-group check, schema cache),
  `_sd_target_block`, `_group_index`, `_resolve_relatedness_labels` (`ids =`
  label matching of `A` works for the sigma-side animal term).
* `_ordinary_laplace_optimize`, `_finite_hessian_from_gradient`,
  `_vcov_from_hessian`, `DrmFit` with `_withnll/_withranef/_withmarginal`.
* Pattern from `_fit_gamma_sigma_sd_regression` (per-group AGHQ) is **not**
  reusable for 6a: it needs independent groups. 6a needs a joint latent
  vector because `id` and `site` cross.

## Cost and size limit

Dense `H` of size `m = G_a + G_b`. Each outer evaluation does one Float64
Newton solve (a few Cholesky factorisations) plus one Dual-valued
factorisation with `p` partials. Measured on one core, steady state:
6a with `G_a=50, G_b=80, n=1600` about 3 s per fit after compilation (first
fit in a session about 20 s of compilation); 6b with 48 groups under 1 s.
Cubic in `m`: `m` is capped at 600 and larger problems are refused by name.
A sparse (Takahashi) path would remove the cap; it is not needed for the
twin tests and is out of scope here.

## Scope and refusals

Refused with an explicit message: REML, missing responses, `meta_V`,
`sd(.., phylogenetic)`, a structured mean effect, a mean random effect without
its own SD formula, mean random slopes, `coef()` selector mismatches, spatial
or phylo sigma effects, `K`/`A`/`coords` given to the ordinary shape,
rank-deficient or within-group-varying SD predictors, `m > 600`, matrix of the
wrong size or not positive definite.

## Risks and gaps

1. **Laplace is an approximation.** Evidence is agreement with an independent
   dense Laplace at the fitted theta, not coverage or bias. Few observations
   per sigma group (this fixture has 20-32) is the regime where Laplace on a
   scale effect is weakest. No claim beyond the likelihood value.
2. **Non-convex inner problem.** Several modes are possible in principle; the
   solver starts from the warm start then from zero. The oracle confirms the
   returned mode is a PD stationary point on every tested fit.
3. **R bridge not wired.** `src/bridge.jl` was not changed; `coef(fit,
   'sd(site, dpar = "sigma")')` naming from R needs the bridge to map the
   hashed block names (it already does for other two-target routes).
4. **`ids =` for `A`** works; `K` has no label option in Julia (same as the
   mean-side route).
5. Joint `mu`+`sigma` with different groups and **no** SD formula (plain
   `(1|id)` + `sigma ~ (1|site)`) is still refused by the old route; it can
   use this engine with `Z = 1` as a follow-up.

## Verdict

* 6a: modest (about 330 lines incl. router). Implemented.
* 6b: modest once the 6a engine exists (a prior precision `K^-1` and a
  one-block latent vector). Implemented, animal and relmat only.
* Order used: engine and 6a first (oracle at 1e-4), then 6b as a
  parameterisation of the same engine.
