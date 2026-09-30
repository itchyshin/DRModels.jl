- **Behaviour change: a degenerate finite-difference Hessian now gives a NaN vcov, not
  fabricated SEs.** `_finite_hessian` (the outer-objective Wald Hessian behind the
  sparse-Laplace GLMM, crossed, phylo/relmat, cumulative, ordinary-Laplace,
  bivariate q=2 structured and sparse-phylo variance-block routes) used to react to a
  non-finite Hessian by warning and adding `1e12` to the diagonal, so the reported SEs
  came from a ridged matrix (near zero, looking precise), and it never treated the
  `1e18` failed-fit objective sentinel as a failure. It now returns an all-NaN
  matrix of the same shape, with a warning, when the objective at the fitted point --
  or any stencil probe -- is non-finite or a `>= 1e16` sentinel, and
  `_vcov_from_hessian` passes a non-finite Hessian through as an all-NaN vcov (the
  NaN-vcov convention of the other guarded Hessian helpers; `stderror` reports `Inf`
  for it, as for `se = false`). The `1e12` ridge is removed. Healthy fits are
  bit-identical (vcov compared to main on Julia 1.10). Only degenerate fits change.
