# CumulativeLogit predict fixtures (#1121)

Copied unchanged from the drmTMB sweep-oct16 lane t6 data, drmTMB commit
`42ca52186e9da6af01280feea599daaa6d8ca93d`, `sweep-oct16/t6/data/` (generator:
`sweep-oct16/t6/R/00_make_data.R`).

- `cumulative_logit_fixed.csv`: n = 240, y in 1..4 from a latent logistic with
  eta = 0.5 x + f-effect and cutpoints (-1, 0.5, 1.8); seed 20261016626
  (R seed 261016626).
- `cumulative_logit_ri.csv`: the same design plus a `(1 | g)` intercept; seed
  20261016627.
- `nd_main.csv`: 18 prediction rows (x in {-1, 0, 1.5} by f in {a, b, c}, for
  a seen group `g01` and an unseen group `gNEW`).

The drmTMB reference values pinned in `test/test_1121_cumulative_predict.jl`
are generated outputs only (no drmTMB source): drmTMB 0.7.1,
`predict(fit, newdata = nd, dpar = "mu", type = "link")` and the delta SE
`sqrt(x' V x)` from `vcov(fit)`, re-run on 2026-10-19 (R 4.6.0, macOS); they
agree with the lane's `results/r_cumulative_logit_*_pred.csv` to about 1e-14.
