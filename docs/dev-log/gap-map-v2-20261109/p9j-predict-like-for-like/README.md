# P9j: compare `predict` like for like in the gap-map v2 harness

Owner answer (Brian consensus, 12 Nov 2026, P9j): fix the harness; add a Julia method only if the R bridge needs it.

What was wrong. `run_r.R` called `predict(fit)` (no `newdata`). In drmTMB that returns the fitted rows and also adds the fitted random effects. `run_julia.jl` calls `predict(fit, d)`, which is population level. Different calls, so five `predict` cells of the K chunk showed `entry:predict:Julia-fails` and six more showed `predict-mean-differs`.

Fix. `run_r.R` now calls `predict(fit, newdata = d)` for the `predict` probe, the same call the Julia runner makes. The old call stays as an informational probe named `predict_insample`.

Result on the 11 K cells (R 4.6.1 with drmTMB 0.8.1; Julia is DRModels main at 525e37423, Julia 1.13.3; `K-cells-before-after.tsv`): every `entry:predict:Julia-fails` and `predict-mean-differs` flag is gone. Where a predictor has a missing value, R now refuses too ("New prediction data contains missing value for required predictor"), as Julia does. No cell changed class. The `entry:simulate:Julia-fails` flag on two cells is a different question and is not touched here.

No new Julia method is needed. The R bridge already gets the in-sample case through `predict_parameters`; see the docstring at `src/bridge.jl` ("Covers the in-sample case (R's `newdata = NULL`)").

Not verified. Only the 11 K cells were rerun, not all 492. `classify.py` is unchanged.
