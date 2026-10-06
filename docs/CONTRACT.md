# DRModels Julia POC bench — file contract

Both the R harness and the Julia harness must agree on these formats.

## Cells

5 cells, defined inline by both harnesses:

| cell_id | n    | model        | mu formula(s)              | sigma formula(s) | rho12 formula |
|---------|------|--------------|----------------------------|------------------|---------------|
| u_small | 100  | univariate   | y ~ x1 + x2                | sigma ~ x1       | —             |
| u_med   | 500  | univariate   | y ~ x1 + x2                | sigma ~ x1       | —             |
| u_large | 2000 | univariate   | y ~ x1 + x2                | sigma ~ x1       | —             |
| b_small | 200  | bivariate    | mu1 = y1 ~ x1, mu2 = y2 ~ x1 | sigma1 ~ 1, sigma2 ~ 1 | rho12 ~ 1 |
| b_med   | 1000 | bivariate    | mu1 = y1 ~ x1, mu2 = y2 ~ x1 | sigma1 ~ x1, sigma2 ~ x1 | rho12 ~ x1 |

True parameter values (used both for simulation and recovery checks):

- **Univariate**: β_mu = (1.0, 0.5, -0.3), β_sigma = (0.2, 0.15) on log link
- **Bivariate**:
  - β_mu1 = (1.0, 0.4), β_mu2 = (-0.5, 0.6)
  - β_sigma1 (b_small): 0.1 (intercept-only); β_sigma1 (b_med): (0.1, 0.2)
  - β_sigma2 (b_small): -0.05; β_sigma2 (b_med): (-0.05, 0.1)
  - β_rho12 (b_small): 0.4 (gives ρ ≈ 0.38); β_rho12 (b_med): (0.4, 0.3)
  - rho link: `ρ = 0.99999999 · tanh(eta)`

Random seed: 42, applied per cell (`set.seed(42 + cell_index)` and the
Julia equivalent — but Julia doesn't read the fixture from RNG, it
reads the CSV the R side wrote, so seed only matters for R/Julia
fixture generation must use the SAME seed).

**To keep R and Julia comparable**: R generates the fixture, writes it
to CSV, then BOTH R and Julia read the CSV and fit. This way both
engines see literally the same numbers.

## File layout

```
drm-julia-poc/
├── CONTRACT.md                 (this file)
├── R/
│   ├── gen_fixtures.R          generates CSV fixtures + truth files
│   ├── fit_r.R                 fits drmTMB on each fixture; writes r_results.json
│   └── compare.R               aggregates + prints summary table
├── julia/
│   ├── Project.toml            ForwardDiff, Optim, Distributions, CSV, JSON3, etc.
│   └── fit_julia.jl            implements Gaussian distributional regression POC;
│                               fits each fixture; writes julia_results.json
├── fixtures/                   <cell_id>.csv per cell (data + meta in filename)
│                               <cell_id>_truth.json (true parameters)
├── results/
│   ├── r_results.json          per-cell R drmTMB fits
│   └── julia_results.json      per-cell Julia POC fits
└── report/
    └── summary.md              wall-clock + Δ logLik + Δ coef table
```

## CSV fixture format

`fixtures/<cell_id>.csv`:

- Univariate cells: columns `y, x1, x2` (always 3 columns)
- Bivariate cells: columns `y1, y2, x1`     (b_small)
                   columns `y1, y2, x1, x2` (b_med — but only x1 in formulas)

Actually keep it simple: bivariate cells have columns `y1, y2, x1, x2`
(four columns always, even if x2 unused). Headers always present.

## Truth JSON format

`fixtures/<cell_id>_truth.json`:

```json
{
  "cell_id": "u_med",
  "n": 500,
  "model": "univariate",
  "beta_mu":    [1.0, 0.5, -0.3],
  "beta_sigma": [0.2, 0.15]
}
```

Bivariate:

```json
{
  "cell_id": "b_med",
  "n": 1000,
  "model": "bivariate",
  "beta_mu1":    [1.0, 0.4],
  "beta_mu2":    [-0.5, 0.6],
  "beta_sigma1": [0.1, 0.2],
  "beta_sigma2": [-0.05, 0.1],
  "beta_rho12":  [0.4, 0.3]
}
```

## Results JSON format

`results/r_results.json` and `results/julia_results.json`:

```json
[
  {
    "cell_id":     "u_med",
    "engine":      "r_drmTMB" | "julia_poc",
    "time_s":      0.123,
    "time_s_med":  0.119,     (median of 5 reruns; for headline timing)
    "logLik":      -712.3,
    "converged":   true,
    "n_iter":      18,        (optional, NA if not available)
    "beta_mu":     [1.02, 0.48, -0.31],
    "beta_sigma":  [0.18, 0.16]
    (univariate ↑; bivariate has beta_mu1/mu2/sigma1/sigma2/rho12)
  },
  ...
]
```

Both engines must run each fit **5 times** and report the median wall-
clock time. (Bash `time` and Julia's first-call JIT cost both bias
single timings; median of 5 is the floor for honest comparison.)

## What counts as a "match"

After both engines produce results, `compare.R` reports:

- |Δ logLik| per cell (gate: < 1e-3)
- max |Δ coef| per cell, relative scale (gate: < 1e-2 for now —
  loose because the POC is not aiming for machine precision)
- median speedup ratio = R_time / Julia_time per cell

Print one line per cell + a headline median speedup ratio.

## Univariate Gaussian likelihood — agreed parameterisation

Both engines fit:

    y_i ~ Normal(mu_i, sigma_i)
    mu_i    = X_mu_i' * beta_mu                       (identity link)
    sigma_i = exp(X_sigma_i' * beta_sigma)            (log link)

drmTMB's gaussian location-scale already follows this; Julia must
match.

## Bivariate Gaussian likelihood — agreed parameterisation

    (y1_i, y2_i) ~ MVN((mu1_i, mu2_i), Sigma_i)
    mu1_i    = X_mu1_i' * beta_mu1                    (identity link)
    mu2_i    = X_mu2_i' * beta_mu2                    (identity link)
    sigma1_i = exp(X_sigma1_i' * beta_sigma1)         (log link)
    sigma2_i = exp(X_sigma2_i' * beta_sigma2)         (log link)
    eta_rho_i = X_rho12_i' * beta_rho12
    rho_i    = 0.99999999 * tanh(eta_rho_i)           (atanh_guarded link)
    Sigma_i  = [sigma1_i^2,           rho_i*sigma1_i*sigma2_i;
                rho_i*sigma1_i*sigma2_i, sigma2_i^2]

drmTMB's biv_gaussian uses the same parameterisation.

## Bootstrap-threading demo (extension)

A focused parametric-bootstrap experiment on 2 cells (u_med, b_med),
n_boot = 199 per cell, comparing four configurations:

| label | engine | parallelism | implementation |
|---|---|---|---|
| `r_serial`   | R/drmTMB        | 1 core                | `for` loop, refit drmTMB each iter |
| `r_parallel` | R/drmTMB        | 10 cores via mclapply | `parallel::mclapply(mc.cores = 10)` |
| `jl_serial`  | Julia POC       | 1 thread              | `for` loop, refit Julia POC each iter |
| `jl_threaded`| Julia POC       | 10 threads            | `Threads.@threads`, `JULIA_NUM_THREADS=10` |

Set `BLAS.set_num_threads(1)` on the Julia threaded path so the outer
threads × inner BLAS threads do not oversubscribe the 20 cores. Same
spirit for `OMP_NUM_THREADS=1` on the R side if the inner TMB build
spawns BLAS threads (unlikely to matter for small problems but cheap
to enforce).

Output: `results/bootstrap_results.json` — one entry per (cell × label)
combination with fields:
- `cell_id`, `label`, `n_boot`, `time_s` (total wall-clock for the
  entire bootstrap), `time_s_per_refit` (= time_s / n_boot), `n_threads`
- A summary row in the final report comparing `r_serial` time vs
  `jl_threaded` time per cell, with the implied total speedup ratio
  (which combines per-refit speedup × thread count × R setup-overhead
  elimination)

The cells reuse the existing fixtures (u_med.csv, b_med.csv) — no new
fixture generation needed.

## Phylogenetic cells (extension — added after the headline 5 cells)

Four additional cells testing univariate Gaussian + a phylogenetic
random intercept on `mu`:

| cell_id | p species | model | mu formula | sigma formula |
|---------|----------:|-------|------------|---------------|
| phylo_p50  | 50   | univariate Gaussian | y ~ x1 + phylo(1 \| species) | sigma ~ 1 |
| phylo_p200 | 200  | univariate Gaussian | y ~ x1 + phylo(1 \| species) | sigma ~ 1 |
| phylo_p500 | 500  | univariate Gaussian | y ~ x1 + phylo(1 \| species) | sigma ~ 1 |
| phylo_p1000 | 1000 | univariate Gaussian | y ~ x1 + phylo(1 \| species) | sigma ~ 1 |

One observation per species. Truth: β_mu = (1.0, 0.5), log σ_eps = -1.2
(σ_eps ≈ 0.30), σ_phy = 0.8 (i.e. log σ_phy = -0.223). Tree generated
via `ape::rcoal(p)` with `set.seed(42 + cell_offset)` where
cell_offset is 100, 101, 102, 103.

### Likelihood (closed-form marginal)

With one obs per species, the marginal model is:

    y ~ MVN(X β, σ²_phy · Σ_phy + σ²_eps · I_p)

where Σ_phy is the p×p ultrametric phylogenetic covariance derived
from the tree (`ape::vcv(tree)` in R; rotation-trick decomposition
optional). Both engines fit this closed form — no Laplace.

### Inter-engine handoff for the tree

R generates the tree, writes:

- `fixtures/<cell_id>.csv` — columns `y, x1, species` (species is
  1-indexed integer; same as row index when one obs per species)
- `fixtures/<cell_id>_sigma_phy.csv` — p×p matrix of Σ_phy with
  numeric headers `s1, s2, ..., sp`. Julia reads this directly. (We
  do NOT pass the Newick tree to Julia; the covariance matrix is the
  sufficient statistic for the closed-form marginal.)
- `fixtures/<cell_id>_truth.json` — adds `sigma_phy: 0.8`,
  `sigma_eps: 0.3` alongside the beta_mu vector. `model: "phylo_uni"`.

### Julia POC implementation

Parameter vector for phylo cells: `[beta_mu (length q), log_sigma_phy, log_sigma_eps]`

```julia
function nll_phylo_uni(theta, y, X_mu, Sigma_phy)
    q = size(X_mu, 2)
    beta_mu = theta[1:q]
    log_sigma_phy = theta[q+1]
    log_sigma_eps = theta[q+2]
    sigma_phy = exp(log_sigma_phy)
    sigma_eps = exp(log_sigma_eps)
    p = length(y)
    Sigma_y = (sigma_phy^2) .* Sigma_phy .+ (sigma_eps^2) .* I(p)
    L = cholesky(Symmetric(Sigma_y)).L
    e = y .- X_mu * beta_mu
    z = L \ e
    logdet_Sigma = 2 * sum(log.(diag(L)))
    return 0.5 * (dot(z, z) + logdet_Sigma) + 0.5 * p * log(2π)
end
```

Init: OLS for beta_mu; `log_sigma_phy = 0` (σ_phy = 1); `log_sigma_eps = -1`.

### Bivariate Gaussian + q=4 location-scale phylogenetic block

**Out of scope for this POC.** This model has phylogenetic random
effects on all four of `mu1`, `mu2`, `sigma1`, `sigma2` sharing one
4×4 phylogenetic covariance block. Because sigma depends on a random
effect non-linearly, the marginal is no longer closed-form — it
requires a Laplace approximation with AD through the inner
mode-finding step. That is the "TMB-like machinery" deferred to
DRModels.jl v0.3+ in the parent plan.

The R side will be timed for reference (drmTMB can fit it via TMB),
but the Julia side will be marked "not implemented in POC" in the
final comparison. This is honest about the current state and clarifies
exactly what work the Laplace-machinery phase would unlock.

### Cell q4_p100 (R-only timing reference, no Julia POC)

| cell_id | p species | model | description |
|---------|----------:|-------|-------------|
| q4_p100 | 100  | bivariate Gaussian + q=4 phylo block | mu1, mu2, sigma1, sigma2 all carry phylo(1 \| species, tree=tree, label="core"); rho12 ~ 1 |

One observation per species. Truth: β_mu1 = (1.0, 0.4), β_mu2 = (-0.5,
0.6), intercepts on sigma1 = 0.1, sigma2 = -0.05; β_rho12 = 0.4 (ρ ≈
0.38); 4×4 phylogenetic covariance Λ_phy diagonal with entries
(0.7, 0.7, 0.3, 0.3) on log scale for sigma slots — i.e. moderate
phylogenetic SD on each of the four parameters, no cross-correlation
for simplicity (use `Λ_phy = diag(0.7, 0.7, 0.3, 0.3) * I + jitter`
or equivalent). R simulates from a known truth; Julia POC does not
fit. compare.R should print "Julia: not implemented (POC scope)"
for this cell.
