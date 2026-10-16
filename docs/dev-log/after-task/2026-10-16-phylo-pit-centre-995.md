# After-task: Gaussian phylo(1 | g) PIT centre (#995, with #996)

## 1. Goal

Owner decision (Option A): centre the Gaussian `phylo(1 | g)` quantile residual on the
fixed-effect mean `Xβ̂`, not on `fit.means[:mu]` (which holds `Xβ̂ + S û`), and fold in
#996 (per-tip prior variance σ_b²·C_ii).

## 2. Implemented

- `src/location_only.jl`: `LocOnlyRanef <: AbstractDict` wraps the existing
  `group => BLUPs` Dict with `mu_fixed = Xβ̂` and the `LocOnlyProblem`; attached on the
  sparse L-BFGS route (ML, REML, penalised) and on the EM route (which stored no BLUPs
  before). `_loconly_row_tip_variance` gives C_ii per row (sparse solves with `Q_cond`).
  `fit.means`/`fit.scales` were not used: `sigma()` and the bridge enumerate them.
- `src/quantile_residuals.jl`: `_ranef_marginal_mix` returns the fixed-effect centre and a
  per-row σ_b√C_ii for these fits; `_cdf_value` accepts a per-row quadrature scale.
  The structured-PIT warning stays for every other structured route.
- Tests: `test/test_995_phylo_pit_centre.jl` pins the residual against the closed-form
  marginal on a height-1 and a height-3 tree, for sparse, REML and EM fits;
  `test_lane_j4_predict_numerics.jl` now expects no warning on the sparse fit and keeps it
  for the dense `algorithm = :gls` fallback.

## 3. Not done / open

- #981 (spatial / relmat / animal structured effects) still warns.
- No local Julia in the authoring environment: verified by CI only.
