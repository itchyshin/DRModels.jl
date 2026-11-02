# After-task: Gaussian closed-form quantile residual (#1103)

## Goal

The remaining half of #1103: a Gaussian (or LogNormal) fit whose quantile residual already uses the closed-form marginal `N(μ, σ² + σ_b²)` must return that z-score, not `Φ⁻¹(clamp(Φ(z), eps, 1-eps))`.

## Implemented

`_quantile_residuals` returns `(y - η) / √(σ² + v_b)` when `_ranef_marginal_mix` reports `:mu_gaussian`, and the log-y form when it reports `:mu_lognormal`. Other continuous families keep the PIT path.

## Evidence

`test/test_quantile_residuals_re_axis.jl` adds a `σ_b/σ > 4` cell that requires the residual to match `z` to 1e-12 and to differ from the round trip by more than 1e-3. Existing axis-routing cells now compare against the exact z-score.

## Not done

The `Φ⁻¹(Φ(z))` round trip remains for every family that is not on this closed-form path (the second half of the issue as originally filed, for non-Gaussian continuous families).

No drmTMB GPL source was used.
