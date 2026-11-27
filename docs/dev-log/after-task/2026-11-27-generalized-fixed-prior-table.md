# Generalized fixed-prior table

The prior table now accepts normalized Student-t densities for fixed-effect coefficients, alongside normal densities and ridge penalties. It retains the existing Gamma rows for Gaussian random-intercept and residual standard deviations. The table parser continues to reject unsupported classes, scales, selectors and distributions. These priors contribute to a penalized maximum-likelihood fit; they do not produce posterior samples or calibrated uncertainty.

## Validation

The DRModels test inventory contains 719 files. The exact partition preflight confirmed 360 files in shard 1 and 359 in shard 2, with no gaps or duplicates, and confirmed that `Pkg.test()` honored the shard filter. On the implementation source `60f83229710a44ac360f21857e6287d17c51837f`, shard 1 passed all 360 files. The original shard 2 run exited 1 because `test/test_api_stability.jl` did not classify the exported `drm_prior_student_t` symbol as experimental.

I replayed the exact same 359-file shard 2 assignment on `f1665b8603ecfd49c51784b3aafdd71e1c8ba346`. That commit adds the missing test classification and changes no production source, Project.toml or Manifest.toml. The replay exited 0 on Julia 1.12.6, and its log ends `Testing DRModels tests passed`. Its before/after source-integrity snapshots match. The original controller failure and passing replay remain separate receipts. Their passing shard results cover all 719 assigned test files across two runs. The initial controller remains red in its preserved receipt.

The implementation source tree is `105ca0b6d06977bac0820d8442fb73b1fd1f9105`. The run receipts are under `/home/snakagaw/hsq_work/generalized-prior-table-arc/drmodels-prior-table-main-60f832297-20261127/`. The original attempt is `full-suite-attempt-04`; the corrected shard 2 replay is `full-suite-attempt-06-api-tier-replay`.

## Integration status

The candidate branch remains local and unpushed. Its `docs/src/priors.md` change overlaps open DRModels PRs #1327, #1328 and #1321, so I have not opened a DRModels PR or changed those branches. The gllvmTMB and GLLVModels.jl draft PRs remain on their validated heads. Their package-suite evidence is complete, but required GitHub checks remain pending or in progress. No PR has been merged and no package has been released.
