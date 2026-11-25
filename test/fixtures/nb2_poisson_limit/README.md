# NB2 at the Poisson limit: the #1039 / #1069 data

Poisson counts with one random intercept: 50 groups x 20 rows, group SD 0.3, intercept
log(150). Columns `y,g`. Simulated in Julia (no external data, nothing from drmTMB).

Generated on Julia 1.12.6 with the #1039 repro's own code (`Random.seed!(seed)`, then
`b = 0.3 .* randn(50)`, then `rand(Distributions.Poisson(exp(log(150) + b[g[i]])))` row by
row; Distributions 0.25.131). The default RNG stream differs on Julia 1.10, which draws
different counts from the same seed (seed 20282004: sum 165491, not 165495), so
`test/test_nb2_poisson_limit.jl` reads these files instead of regenerating them.

| file | seed | n | sum(y) | md5 |
|---|---|---|---|---|
| `data_seed20282003.csv` | 20282003 (#1069) | 1000 | 152384 | `e83c2d6301c5b8402b8bcdf32af2ac0a` |
| `data_seed20282004.csv` | 20282004 (#1039) | 1000 | 165495 | `55f5a8ef61db5a07d8e0d5cddd93993a` |

The seed-20282004 checksums match the ones recorded in #1039.
