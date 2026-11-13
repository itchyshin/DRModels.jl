# P9j: compare `predict` like for like in the gap-map v2 harness

Owner answer (Brian consensus, 12 Nov 2026, P9j): fix the harness; add a Julia method only if the R bridge needs it.

What was wrong. `run_r.R` called `predict(fit)` (no `newdata`). In drmTMB that returns the fitted rows and also adds the fitted random effects. `run_julia.jl` calls `predict(fit, d)`, which is population level. Different calls, so five `predict` cells of the K chunk showed `entry:predict:Julia-fails` and six more showed `predict-mean-differs`.

Fix. `run_r.R` now calls `predict(fit, newdata = d)` for the `predict` probe, the same call the Julia runner makes. The old call stays as an informational probe named `predict_insample`.

Result on the 11 K cells (R 4.6.1 with drmTMB 0.8.1; Julia is DRModels main at 525e37423, Julia 1.13.3; `K-cells-before-after.tsv`): every `entry:predict:Julia-fails` and `predict-mean-differs` flag is gone. Where a predictor has a missing value, R now refuses too ("New prediction data contains missing value for required predictor"), as Julia does. No cell changed class. The `entry:simulate:Julia-fails` flag on two cells is a different question and is not touched here.

No new Julia method is needed. The R bridge already gets the in-sample case through `predict_parameters`; see the docstring at `src/bridge.jl` ("Covers the in-sample case (R's `newdata = NULL`)").

Full regeneration (second commit). The whole R side (all 492 cells, one run of about one minute on Totoro with 3 processes) was rerun with the fixed `run_r.R`, and `classify.py` (unchanged) was run against the committed Julia raw files. Check first: `classify.py` on the OLD committed raw files reproduces the committed `maps/*` byte for byte. After the rerun: `raw/r/*.out` and three map files changed (`gap_map.tsv`, `disagreements.tsv`, `flagged_agreeing.tsv`); class, sub-class and the R outcome are the same in all 492 cells. 25 cells change, listed in `all-cells-before-after.tsv`: in 12 the flags change (all removals, none added: 9 K cells, `D|binomial-cbind|(1+x1|g)|link=probit`, `...|link=cloglog`, and `H|meta|response with NA`), and in 13 more only the `pred_agree` column goes from False to True (cells with random effects, where R's old call added the fitted effects). The Julia raw files were not rerun: `run_julia.jl` is unchanged.

Not verified. The Julia side was not rerun, so any Julia change since the committed raw files is not in these maps.
