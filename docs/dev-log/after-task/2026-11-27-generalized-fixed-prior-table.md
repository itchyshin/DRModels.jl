# Generalized fixed-prior table

This report records the 27 November candidate and its validation. The current combined candidate has since changed: its live R-to-Julia fixtures passed on a later source, and its latest Julia package suite is in progress. See PR #1631 for the current source and results. The bridge limitation below describes the earlier candidate.

The prior table now accepts normalized Student-t densities for fixed-effect coefficients, alongside normal densities and ridge penalties. It retains the existing Gamma rows for Gaussian random-intercept and residual standard deviations. The table parser continues to reject unsupported classes, scales, selectors and distributions. These priors contribute to a penalized maximum-likelihood fit; they do not produce posterior samples or calibrated uncertainty.

## Validation

The DRModels test inventory contains 719 files. The exact partition preflight confirmed 360 files in shard 1 and 359 in shard 2, with no gaps or duplicates, and confirmed that `Pkg.test()` honored the shard filter. On the implementation source `60f83229710a44ac360f21857e6287d17c51837f`, shard 1 passed all 360 files. The original shard 2 run exited 1 because `test/test_api_stability.jl` did not classify the exported `drm_prior_student_t` symbol as experimental.

I replayed the exact same 359-file shard 2 assignment on `f1665b8603ecfd49c51784b3aafdd71e1c8ba346`. That commit adds the missing test classification and changes no production source, Project.toml or Manifest.toml. The replay exited 0 on Julia 1.12.6, and its log ends `Testing DRModels tests passed`. Its before/after source-integrity snapshots match. The original controller failure and passing replay remain separate receipts. Their passing shard results cover all 719 assigned test files across two runs. The initial controller remains red in its preserved receipt.

The implementation source tree is `105ca0b6d06977bac0820d8442fb73b1fd1f9105`. The run receipts are under `/home/snakagaw/hsq_work/generalized-prior-table-arc/drmodels-prior-table-main-60f832297-20261127/`. The original attempt is `full-suite-attempt-04`; the corrected shard 2 replay is `full-suite-attempt-06-api-tier-replay`.

The required engine regression benchmark also passed on the same tested code source (`f1665b8603ecfd49c51784b3aafdd71e1c8ba346`). It exited 0 in 21 seconds on one thread. The fit converged at logLik −256.5066, 0.0034 from the −256.51 baseline. The before/after source snapshots matched for HEAD, tree, Project.toml, Manifest.toml and clean Git status. Its log is `/home/snakagaw/hsq_work/generalized-prior-table-arc/drmodels-prior-table-main-60f832297-20261127/engine-benchmark-f1665-20261127/run.log` (SHA-256 `576230b478010a88e986e427c7dfe83d961f135661afc5c1d6c3981aa6e1d290`).

## Integration status

The candidate's `docs/src/priors.md` includes the existing prior guidance and overlaps open DRModels PRs #1327, #1328 and #1321. Reconcile that section during review; this candidate does not change those branches. The R-to-Julia bridge remains unvalidated because the embedded Julia startup failed in the test environment. No merge or release is authorized by this report.
