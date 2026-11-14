# Unified prior-table adapter

The adapter accepts Tables.jl-compatible inputs for fixed-effect normal and ridge penalties and Gamma densities on one or two named Gaussian mean random-intercept SDs. It translates the table into existing prior specifications. The table Gamma form takes mean and shape; the constructor takes shape and rate. Random slopes, correlations, structured effects and unsupported families remain outside this slice.

Blank fixed-effect selectors cover non-intercept coefficients. Blank SD selectors cover supported random-intercept SDs. Overlapping selections and coefficients literally named `all` are rejected. The original table is retained, and update replays the resolved targets.

Independent review found that a mixed table could re-expand a fixed-effect wildcard on update; `0c527f07d` froze the selected targets. The full Julia suite on that commit then exposed a no-prior joint-fit dispatch regression; `e4fee23c8` corrected it. Focused joint-model, bootstrap, table, oracle and live-bridge tests passed on the corrected code. The earlier failures and covariance warnings are retained in the validation evidence.

The dense oracle tests covered 150 objective, gradient and Hessian points and 12 fitted comparisons per package. Test-only helpers reconstructed R-marshalled bridge payloads for 150 two-intercept and 30 single-intercept points; these helpers bypass public bridge result serialization. The live fitted-result bridge passed 38 checks at R `d5bd21009` and Julia `e4fee23c8`.

The e4fee23c8 full suite found a missing direct test dependency and a Poisson offset comparison failure that reproduced on the foundation. Test-only commit 164068f4f added Tables to the test environment and recalibrated the Poisson fitted-value comparison while adding offset and oracle checks; the underlying optimizer difference remains unexplained. A 53-assertion focused Pkg.test sandbox passed. All four corrected Linux Pkg.test shards passed on 164068f4f. The logs contain warnings and skips, including R parity and selected slow tests; these results do not establish cross-platform or complete parity coverage. A synthetic combination with Julia main `8d0bd9ea5` passed focused joint-model, table, oracle and sigma-random-effect tests. That snapshot is not a merged branch or a full current-main suite.

This is penalized maximum likelihood and does not provide posterior samples or calibrated uncertainty. The R and Julia branches remain drafts; no merge or release follows from these checks.
