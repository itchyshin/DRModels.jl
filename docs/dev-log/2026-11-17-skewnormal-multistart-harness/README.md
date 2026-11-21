# SkewNormal parity cells: call R with `multi_start = 5`

Date: 2026-11-17. Julia: main 29bb6074c, Julia 1.13.3. R: drmTMB 0.8.2.

## Why

The map 2 cell `F|skew|response=y_g(symmetric)` was `open` (log-likelihood gap 5.4e-02).
drmTMB's default single start stops at a local maximum (-243.9303305893). Julia reaches
-243.8767839692, and so does R when it is given `drm_control(multi_start = 5)`. The gap is a
property of R's default start, not a Julia defect.

## What changed in the harness

`run_r.R` (the gap-map v2 R runner, kept on the server, not in this repository) now adds
`control = drm_control(multi_start = 5L)` to every cell whose R call contains
`skew_normal()` and has no random effect and no `control =` already. `run_r.R.patch` is the
diff. No cell definition, tolerance, classifier rule or Julia call changed.

Random-effect SkewNormal cells (`(1 | g)`) are left on the default start, so their recorded
values do not move. The two cells R refuses (`REML`, `(1|g/h)`) are unchanged.

## Result

Only the 15 fixed-effect SkewNormal cells that R fits were rerun (16 s). Table:
`skewnormal_multistart_results.tsv`. R's log-likelihood is identical to the single-start value
to 10 digits in 14 cells. The only cell that moves is the target one, from -243.9303305893 to
-243.8767839692, and it now agrees with Julia (log-likelihood and fitted values).

Class for `F|skew|response=y_g(symmetric)`: **agree at the global optimum** (R with
`multi_start = 5`; single-start R stops at a local maximum). When the ledger of #1388 is
rebased, that row changes from `open` to `agree` with this note. No test expectation changes:
`test/test_skewnormal_restarts.jl` keeps the recorded single-start drmTMB value as history.
