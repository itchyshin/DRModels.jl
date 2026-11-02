# After-task: redraw sigma ~ (1 | g) in the bootstrap simulator (#926)

## Goal

The remaining half of #926: `r2_constant_sigma` already refuses `sigma ~ (1 | g)`, but the marginal bootstrap simulator still treated residual SD as constant.

## Implemented

`_sigma_ranef_components` reads an ordinary `sigma ~ (1 | g)` term from the formula and the fitted log-σ SD. `_sigma_only_marginal_simulator` covers the no-mean-RE route (previously `nothing`, so bootstrap used `simulate` at b = 0). The Gaussian mean-RE draw multiplies the residual SD by `exp(u_g)` when that term is present.

## Evidence

`test/test_sigma_re_bootstrap_926.jl`: sigma-only and mean-plus-sigma cells. Each requires the simulator to exist and the spread of per-group residual SDs across draws to be larger than constant-σ sampling noise.

## Not done

A structured (phylo/relmat/animal/spatial) random effect on `sigma` is still refused / not drawn. The LSS `sd(g) ~` route is unchanged.

No drmTMB GPL source was used.
