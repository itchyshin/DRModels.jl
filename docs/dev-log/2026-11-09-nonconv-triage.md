# Triage of the wave-10 non-converged cells, 2026-11-09

Reader: whoever owns the R-Julia parity lane. Source: rank 4 of `docs/dev-log/2026-11-09-gap-map-wave10.md` (branch `claude/gap-map-wave10-20261109`), five non-converged cells plus one timeout. Base: `0bfcde2bb` (PR #1272). R reference: drmTMB 0.8.1 on Totoro (`~/drm_campaign_20261108/Rlib`), same data and formulas as the wave-8 fuzz harness. All Julia numbers are Julia 1.10.12, one thread.

## Verdicts

| Cell | Before (Julia, R logLik) | Root cause | Verdict | After |
| --- | --- | --- | --- | --- |
| `gamma\|phylo\|q1\|sd-z` | -621.451975, R -621.451975, flag false | Fminbox L-BFGS stopped with a zero step (`x` unchanged, 3 outer iterations). Mean gradient 1.01e-6 per observation against the 1e-6 reporting limit (`_laplace_outer_converged`); only the sigma coordinate carried it (4.1e-4, the same value at finite-difference steps 1e-5 and 2.9e-4, so a real slope, not noise) | Fixed: Newton polish, flag test unchanged | converged, -621.4519749317 (R -621.4519749318), gradient/n 5e-12 |
| `gamma\|indep\|q2\|sd-none` | -555.596, R -546.343 (9.25 below), flag false | The inner Newton solve of `_fit_indep_re_laplace` reaches step 1.4e-8 and gradient 2.6e-7, then the line search (`Jt <= J0`, decided by rounding at J = 484) rejects every step and halves it to nothing for 200 iterations; the solve is reported failed and the objective returns 1e18 at a benign point (t = 0.3 on the line from the old stop to R's estimate), from the warm start and from zero. BFGS then fails its line search at a point with gradient norm 92. The objective is the same as R's: Julia's `nll` at R's estimate is 546.3427321265 | Fixed: stop the inner solve at the Newton-decrement rounding floor | converged, -546.3427321265 (R -546.3427321263), estimates within 1e-3 of R |
| `poisson\|spatial\|q1\|sd-none` | NaN, flag false, 1 iteration | `_poisson_spatial_marginal` warm-starts the inner mode from `bref`, which is refreshed only on success. The first L-BFGS trial is 28 units long (gradient 28 at the start), succeeds with an extreme mode, and every later evaluation, including the points next to the start, fails the capped Newton loop and returns 1e18; the line search fails with zero gradient | Fixed: retry the inner mode once from b = 0 (as the Gamma structured route does) | converged, -674.4649 in 17 iterations. Not an R disagreement: drmTMB fixes the range at the median positive pairwise distance (5 parameters, -675.6946875), Julia estimates it (6 parameters). With Julia's range fixed at 3.2563 the optimum is -675.6946979 (1e-5 from R); the free range gains 1.23 |
| `nb2\|animal\|q2\|sd-z` | -694.4688, R -694.6354, flag false, 151 iterations | Both fits run to a degenerate ridge. The intercept SD is the same in both (log SD -0.574 + 0.470 z against R's -0.565 + 0.452 z). The slope SD collapses: R reaches log SD -10.84 - 1.33 z; Julia's coefficients run to -126.8 and -125.8, so the slope SD is e^-1 only at the single group with z = -1 and zero elsewhere. That spike is why Julia sits 0.167 above R; it is not a better model. drmTMB reports `degenerate` (Hessian singular, gradient 3e-7); the plain NB2 without the animal term scores -733.88, so the intercept field is real | Keep fit-and-flag; nothing to fix | unchanged, `converged = false` |
| `binomial\|ordinary\|q3\|sd-z` | -248.693, R -248.909, flag false, 1000 iterations (80 s) | Same structure in both: intercept SD coefficients -5.99 + 7.65 z (R -5.87 + 7.46 z), x1 SD -0.61 + 0.64 z (R -0.67 + 0.68 z), x2 SD -2.99 + 4.03 z (R -3.01 + 4.02 z); the three correlation parameters run to -9, 11 and -2.7 (R -10, 15.9, -10), that is correlations at the boundary. Julia's Hessian is not positive definite and its Wald SEs are withheld; R has `pdHess = FALSE` and NA SEs | Keep fit-and-flag; nothing to fix | unchanged |
| `binomial\|ordinary\|q4\|sd-z` | timeout at 300 s, R not converged | Not investigated: q = 4 correlated AGHQ with the same ridge as q = 3 (R has `pdHess = FALSE`, `is_converged = FALSE`). Recorded only | Record only | unchanged |

## Decision for the two cells where R does not converge

Julia fits and flags (`converged = false`, SEs withheld by the existing guards). Reasons: (1) refusing would hide a fit that drmTMB also returns, with the same estimates to about 10 percent on the SD coefficients; (2) the flag is honest, since both fits end on a ridge with a boundary correlation or a one-group SD spike; (3) Julia's higher log-likelihood on the NB2 cell comes from the spike, so it should not be quoted as a better optimum. Not done: a boundary-aware reporter that names the degenerate coordinate (runaway SD coefficient, correlation at the boundary) the way R labels `degenerate`. That is a policy change to the shared flag, not a triage fix.

## What changed

- `src/gamma.jl`, `_fit_gamma_structured_sd_laplace`: when `_laplace_outer_converged` fails and no bound is active, up to two Newton steps on `_finite_hessian`, each kept only if the objective does not rise. The reporting test and its 1e-6 limit are unchanged.
- `src/indep_slopes_laplace.jl`, `inner_mode`: return as converged, after the step, when `0.5 * grad' * step <= 1e-13 * (1 + |J0|)`. This route also serves `Student()` independent terms.
- `src/poisson.jl`, `_poisson_spatial_marginal`: on a failed inner mode, one retry from zero.
- `test/test_nonconv_triage_wave10.jl` with data in `test/fixtures/nonconv_triage/` (tree and coordinates shared with `fixtures/gamma_sdz`). Each cell fails on the base commit (checked on `0bfcde2bb`, see the PR for the output).

## Not verified

- The Newton decrement threshold (1e-13 relative) is a rounding-floor argument, not tuned against other Student cells; the full suite is the evidence.
- NB2 and Binomial q = 3: R's estimates were read from a single R run each; Julia's objective at R's estimate was not evaluated on those two cells.
- The Binomial q = 3 gradient at Julia's stop is 5.8e-6 per observation at the route step, six times the reporting limit after 1000 iterations; whether more iterations would pass was not tried.
- Binomial q = 4 timeout: not rerun.
