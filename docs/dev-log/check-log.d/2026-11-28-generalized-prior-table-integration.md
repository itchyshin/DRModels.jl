# Generalized prior-table integration checks

The DRModels.jl suite covered 731 unique assigned files on production source `d8ccb7195de5fe0ed50f094460a07fc55530ebd7`. Six original sections exited 0. Original sections 2 and 5 failed on test expectations; their exact assignments passed on test-only corrections `2f2effd7521e8b7521fdb66627fbde04085bc5e0` and `f25fa78c387826696ed33767fb85fe3ebf99b3df`. All source snapshots matched. The original failures are preserved, and this is reconciled coverage across runs rather than one passing controller. The original controller-summary print was malformed; individual source-integrity snapshots, logs and exit receipts were checked.

The combined receipt is `/home/snakagaw/hsq_work/generalized-prior-table-arc/drmodels-v5-full-d8ccb719-20261128/attempt-01/combined-source-reconciliation.json`, SHA-256 `eede4bdb0a9f539a8520d0ba9194a336b788f1209a80981eb760d8e1005e4a69`. The R-to-Julia fixture groups passed 47, 12 and 17 assertions on the same production source. Two generated cases reported singular Hessians; these checks do not establish calibrated uncertainty. The isolated prior guide rendered on this source with missing-index, omitted-docstring, navigation and inventory warnings.

GitHub checks and review remain pending. This evidence does not authorize a merge or release.
