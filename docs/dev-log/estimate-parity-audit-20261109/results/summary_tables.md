cells 90; clean 81; boundary/flat 8; Julia-unconverged-while-R-ok 1

### By family (clean cells)

| Group | Cells | max abs logLik diff | max abs coef diff | max abs (SE ratio − 1) | max abs SD diff | max abs pred link diff | max abs pred response diff | max abs CI bound diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beta | 8 | 6.7e-09 | 2.0e-07 | 1.5e-04 | 5.2e-09 | 3.7e-08 | 7.0e-09 | 8.4e-05 |
| binomial | 3 | 1.0e-09 | 1.3e-07 | 0.00316 | n/a | 7.8e-08 | 1.7e-08 | 0.00354 |
| gamma | 9 | 7.0e-09 | 9.4e-08 | 0 | 1.5e-10 | 1.7e-07 | 4.2e-07 | 2.2e-06 |
| gaussian | 34 | 1.0e-09 | 1.3e-08 | 0.00123 | 9.9e-10 | 2.2e-08 | 2.2e-08 | 1.6e-06 |
| lognormal | 5 | 1.0e-09 | 6.8e-10 | 0 | 3.2e-10 | 7.0e-11 | 2.72 | 1.5e-07 |
| nb2 | 6 | 8.0e-09 | 1.1e-07 | 2.0e-05 | 2.9e-08 | 2.0e-07 | 2.2e-07 | 2.2e-05 |
| poisson | 10 | 1.0e-09 | 1.5e-06 | 0 | 1.8e-07 | 1.7e-06 | 3.0e-06 | 2.1e-06 |
| student | 3 | 0 | 6.0e-11 | 0 | 2.9e-11 | 3.9e-11 | 3.9e-11 | 1.2e-06 |
| tweedie | 3 | 0 | 6.0e-08 | 0 | n/a | 1.6e-08 | 3.0e-08 | 2.1e-06 |

### By route (clean cells)

| Group | Cells | max abs logLik diff | max abs coef diff | max abs (SE ratio − 1) | max abs SD diff | max abs pred link diff | max abs pred response diff | max abs CI bound diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| animal | 9 | 2.0e-09 | 1.3e-07 | 3.0e-05 | 1.0e-12 | 8.5e-09 | 1.5e-08 | 4.0e-05 |
| crossed | 10 | 8.0e-09 | 1.5e-06 | 0.00316 | 1.8e-07 | 1.7e-06 | 2.72 | 0.00354 |
| indep | 8 | 3.0e-09 | 1.1e-07 | 1.0e-05 | 2.9e-11 | 3.0e-08 | 1.6e-07 | 1.5e-05 |
| ordinary | 10 | 0 | 1.2e-07 | 2.0e-05 | 0 | 6.9e-08 | 1.84 | 2.2e-05 |
| phylo | 17 | 1.0e-09 | 1.4e-07 | 0.00123 | 5.0e-09 | 1.3e-07 | 1.63 | 5.2e-05 |
| relmat | 16 | 3.0e-09 | 9.4e-08 | 1.5e-04 | 7.0e-12 | 3.3e-08 | 1.31 | 8.4e-05 |
| slope | 3 | 0 | 1.0e-11 | 0 | 2.0e-12 | 2.0e-12 | 1.99 | 1.5e-07 |
| spatial | 8 | 0 | 2.0e-07 | 0 | n/a | 2.0e-07 | 4.2e-07 | 2.2e-06 |

### By estimator (clean cells)

| Group | Cells | max abs logLik diff | max abs coef diff | max abs (SE ratio − 1) | max abs SD diff | max abs pred link diff | max abs pred response diff | max abs CI bound diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ML | 67 | 8.0e-09 | 1.5e-06 | 0.00316 | 1.8e-07 | 1.7e-06 | 2.72 | 0.00354 |
| REML | 14 | 1.0e-09 | 1.3e-08 | 0.00123 | 9.9e-10 | 2.2e-08 | 2.2e-08 | 2.3e-07 |

### Excluded cells

| Group | Cells | max abs logLik diff | max abs coef diff | max abs (SE ratio − 1) | max abs SD diff | max abs pred link diff | max abs pred response diff | max abs CI bound diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J unconverged, R ok | 1 | 1.0e-09 | 3.6e-06 | n/a | 1.8e-07 | 2.7e-07 | 3.3e-07 | n/a |
| boundary/flat | 8 | 2.6e-06 | 6.62 | 5.0e+13 | 3.0e-04 | 1.8e-07 | 4.4e-07 | 4.4e+04 |

Quantiles on clean cells (median, 90th, max):
  coef_max_absdiff: n=81, median 2.5e-09, p90 9.4e-08, max 1.5e-06
  se_ratio_m1: n=81, median 0, p90 0, max 0.00316
  sd_max_absdiff: n=21, median 1.4e-11, p90 5.2e-09, max 1.8e-07
  pred_link_maxabs: n=81, median 2.4e-09, p90 5.0e-08, max 1.7e-06
  ci_max_absdiff: n=81, median 7.5e-08, p90 2.1e-06, max 0.00354
  d_logLik: n=81, median 0, p90 1.0e-09, max 8.0e-09

Simulation: both marginal draws available in 41 of 90 cells
  sim_rel_grand_mean: median 0.00561, p90 0.0287, max 0.0342
  sim_rel_grand_var: median 0.00622, p90 0.252, max 1.05
  sim_rel_mean_perobs_var: median 0.00929, p90 0.312, max 1.15
  sim_abs_p_zero: median 0, p90 0.00101, max 0.0021
  coverage R-ok/J-ok: Counter({('ok', 'ok'): 41, ('ok', 'err'): 22, ('err', 'ok'): 19, ('err', 'err'): 8})

### Fitted values and quantile residuals by route (continuous families; R conditional vs Julia as documented)

| Route | SD formula | Cells | fitted agree (<1e-6) | quantile residual agree (<1e-6) | median corr where not |
| --- | --- | --- | --- | --- | --- |
| animal | no | 1 | 0 | 0 | 0.756 |
| animal | yes | 6 | 0 | 5 | 0.838 |
| crossed | no | 7 | 0 | 0 | 0.789 |
| indep | no | 5 | 0 | 0 | 0.753 |
| indep | yes | 2 | 0 | 0 | 0.668 |
| ordinary | no | 2 | 0 | 1 | 0.87 |
| ordinary | yes | 4 | 0 | 3 | 0.781 |
| phylo | no | 6 | 2 | 0 | 0.853 |
| phylo | yes | 7 | 0 | 4 | 0.853 |
| relmat | no | 4 | 0 | 2 | 0.848 |
| relmat | yes | 6 | 0 | 4 | 0.914 |
| slope | no | 1 | 0 | 0 | 0.968 |
| slope | yes | 2 | 0 | 0 | 0.966 |
| spatial | no | 1 | 0 | 1 | - |
| spatial | yes | 5 | 0 | 4 | 0.957 |
