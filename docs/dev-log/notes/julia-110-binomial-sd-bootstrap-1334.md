# Julia 1.10 — Binomial SD correlated bootstrap refit (#1334)

**Symptom (reproduced on macOS arm64, Julia 1.10.12):** `test/test_binomial_sd_bootstrap_nq_1334.jl` errors on `bootstrap_result(..., B = 1, rng = MersenneTwister(30))` with “refit did not converge or landed on a degenerate optimum” (replicate seed `0x4e804fe864216b88` under Julia 1.10’s `MersenneTwister` stream).

**Root cause:** numerical path, not RNG fixture data. Correlated Binomial SD AGHQ refits used `_optimize_with_fallback` with default Hager–Zhang line search only. On Julia 1.10 / certain Optim.LineSearches stacks, LBFGS can stop without `g_converged` (and occasionally after a rejected line search — see `optim_minimum_guard.jl`). Bootstrap refits always cold-started from the generic `theta0`, so fragile draws failed `_converged_nondegenerate` while the seed fit succeeded.

**Fix:** After a non–gradient-converged Hager–Zhang run, retry BackTracking from the default start and the returned minimizer (`_optimize_with_fallback` in `binomial.jl`). A `theta_warm` bootstrap path was tried and reverted: warm-starting from the seed fit’s `theta` broke intercept-only Binomial SD bootstrap on Julia 1.13 (e.g. `boot_i` in `test_binomial_sd_regression.jl`).

**Regression tests:** `test/test_binomial_sd_bootstrap_corr_refit_1334.jl` mirrors the official `B = 1` bootstrap assertion.