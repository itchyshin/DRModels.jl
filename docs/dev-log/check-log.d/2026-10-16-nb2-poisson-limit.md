# NB2 at the Poisson limit (#1072, #1069, #1039)

- Change (first round): `src/negbinomial.jl` — `_fit_negbin2` and `_fit_negbin2_ranef` now optimise through
  `_nb2_minimize` (AssertionError from the LBFGS/Hager-Zhang line search falls through to
  BackTracking, then Nelder-Mead; a fit on the Poisson plateau, log sigma < -8, is refit once from
  log sigma = -4 and the interior fit is kept only if it has the lower nll) and return through
  `_warn_nb2_poisson_limit` (`src/boundary_diagnostics.jl`), which warns that the fit is at the Poisson
  limit. `converged` is unchanged.
- Change (second round, Code Reviewer block): the sparse-Laplace NB2 routes are covered too. They clamp
  the size at exp(8) (log sigma >= -4), so there the threshold is that clamp. `_nb2_laplace_refit`
  (dispatched on the NB2 kernels, a no-op for every other family) runs inside the three nuisance spines
  in `src/sparse_laplace_glmm.jl` (general-mean: phylo `sigma ~ 1`, relmat/animal/spatial; phylo
  covariate sigma; crossed), right after the first optimise, and refits once from log sigma = -1 when
  every row is at the clamp. `_warn_laplace_clamp` gives an NB2 fit at the clamp the Poisson-limit
  warning instead of the generic #959 line. The crossed `marginal = :AGHQ` route runs its Laplace start
  silently, so it warns once. Both refits compare the nll evaluated at each minimizer, not
  `Optim.minimum` (`src/optim_minimum_guard.jl`).
- Tests: `test/test_nb2_poisson_limit.jl` (red on `727b003`, base + test only). Second round: every
  sparse-Laplace testset is red on `ff0f9ee` (20 failed, 1 errored; the first-round tests pass there);
  the file passes 75/75 on Julia 1.12.6 (default flags, and `--check-bounds=yes` as `Pkg.test` runs)
  and on Julia 1.10.10 (`--check-bounds=yes`). The #1039 / #1069 data are fixtures
  (`test/fixtures/nb2_poisson_limit/`, md5 of seed 20282004 = the one in #1039): `Random.seed!` draws
  different counts on Julia 1.10.
- Local runs (Julia 1.12.6, Optim 1.13.3): 16 touched-route test files, 1,958 tests, pass; the 8 checked
  fits that do not keep a refit are bitwise identical to `ff0f9ee` (theta, logLik, converged, vcov with
  `se = true`);
  `bench/run_sparse_tmb_nd.jl` logLik -256.5123.
- Not done: location SEs with phi held at the limit (#1072 request 3) — needs `_vcov_from_hessian`
  (the SE path), left for the #1073 stack. NB2 routes with neither the warning nor the refit: the
  correlated-slope `(1 + x | g)`, zero-inflated, hurdle, truncated, VA, `marginal = :Laplace` (it has its
  own plateau refit), coupled location-scale and mixed-family fitters.
