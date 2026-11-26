## Slice

Warn on finite `loglik` / `aic` / `bic` / `deviance` when `!is_converged` (refs #1542, #1543).

## Changes

- `_unconverged_status_infocrit_warn` in `comparison.jl`; wired into loglik, aic, bic, deviance, aicc (via aic).
- `convergence_status` reads `temporal()` interpretability limits as `:boundary`.
- Bivariate Student-t: `_vcov_from_hessian_checked`, `_ctl_se`, `_withuncertainty`, PD gate on `converged`.
- Wald `confint` warns on unbounded endpoints.

## Verification

- `julia --project=. test/test_unconverged_infocrit_1542.jl`
- `julia --project=. test/test_convergence_status.jl`
- `julia --project=. test/test_sentinel_fit_level.jl`