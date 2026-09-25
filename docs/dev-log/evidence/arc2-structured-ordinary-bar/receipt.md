# Arc 2 receipt: a structured marker plus an ordinary random effect (`structured_with_ordinary_bar`)

**Question.** Does DRModels.jl fit the same model as native drmTMB
(`engine = "tmb"`) for a Gaussian mean with one structured marker
(`phylo`/`relmat`/`animal`, or `spatial` sent as `relmat` + fixed `K`) and one
or more ordinary random effects?

**Gap before this change (DRModels.jl `da8b3f871`).** The dispatcher sent
`y ~ x + <marker>(1 | ...) + (1 | h)` to the single-structured fitter, which
fitted the marker alone. The ordinary `(1 | h)` term was dropped with no
warning. drmTMB's Arc 1 sweep measured this through the bridge: df 4 and
logLik -236.758989, identical to the marker-only fit, against native df 5 and
logLik -216.285083. The drmTMB bridge therefore refuses the shape under gate
`structured_with_ordinary_bar`.

**Change.** A new route, `_drm_gaussian_structured_plus_ranef` →
`_fit_structured_ranef_gaussian` (`src/gaussian_structured.jl`), dispatched in
`drm(::DrmFormula, ::Gaussian)` (`src/gaussian_core.jl`) before any
single-structured fitter. It fits drmTMB's model: independent blocks, with
`u_k ~ N(0, σ_k² K_k)`, `K = I` for an ordinary bar, and the exact Gaussian
marginal `V = D + Σ_k σ_k² Z_k K_k Z_kᵀ` (dense, ML).

## How to reproduce

```sh
cd docs/dev-log/evidence/arc2-structured-ordinary-bar
Rscript native-fits.R ~/local-scratch/lanes/drmTMB-arc1-pr1304-fold      # ~5 s
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  julia --project=<this DRModels.jl checkout> julia-fits.jl              # ~55 s
```

`native-fits.R` builds the fixtures, fits them natively, and writes the rows,
trees, and `K`/`A` matrices to `fixtures/`, plus `native.tsv`.
`julia-fits.jl` fits the same files through `drm_bridge`, the entry point
drmTMB's R bridge calls. It uses the formula string and the `tree`/`K`/`A`
keyword arguments the bridge sends. For `spatial_h`, native's own fixed-range
exponential `K` from `drm_spatial_coords_precision()` is sent as `relmat`, as
the bridge does. The script writes `julia.tsv` and `comparison.tsv`. It stops
if any native parameter has no Julia twin.

Native reference: drmTMB `709efbeb0` (lane `drmTMB-arc1-pr1304-fold`,
origin/main + Arc 1 evidence files, DLL built). Julia: this branch.

## Result: the same model on all nine fixtures

| fixture | formula (Julia side) | df (both) | native logLik | Julia logLik | abs ΔlogLik | max rel Δ estimate |
|---|---|---|---|---|---|---|
| phylo_h_seed11 | `y ~ x + phylo(1 \| sp) + (1 \| h); sigma ~ 1` (20 tips x 6) | 5 | -136.6486012270 | -136.6486012269 | 1.2e-10 | 6.8e-11 |
| phylo_h_seed22 | same, other seed and shape (30 tips x 4) | 5 | -102.7517002055 | -102.7517002054 | 1.4e-10 | 1.3e-10 |
| phylo_h_sigmax | same with `sigma ~ x` | 6 | -143.4417813812 | -143.4417813811 | 6.6e-11 | 1.0e-10 |
| phylo_sp_same_group | `y ~ x + (1 \| sp) + phylo(1 \| sp)` (no `sd()`) | 5 | -159.9437095358 | -159.9437095357 | 9.5e-11 | 1.1e-10 |
| phylo_h_g | `phylo(1 \| sp) + (1 \| h) + (1 \| g)` | 6 | -132.0247349877 | -132.0247349878 | 9.9e-11 | 1.2e-10 |
| phylo_slope_h | `phylo(1 \| sp) + (0 + x \| h)` | 5 | -128.7684146938 | -128.7684146939 | 1.2e-10 | 4.0e-10 |
| relmat_h | `relmat(1 \| id) + (1 \| h)` | 5 | -154.4255293185 | -154.4255293185 | 1.2e-11 | 1.3e-10 |
| animal_h | `animal(1 \| id) + (1 \| h)` | 5 | -140.8348471281 | -140.8348471281 | 7.4e-12 | 8.3e-11 |
| spatial_h | native `spatial(1 \| site, coords)`, Julia `relmat(1 \| site)` + native's `K` | 5 | -125.6369393640 | -125.6369393640 | 2.3e-11 | 9.3e-11 |

"Estimate" covers every mean and sigma coefficient and every random-effect
SD. Native SDs come from `fit$sdpars$mu`; Julia SDs are `exp` of the `resd_*`
coefficients. Row-level values are in `comparison.tsv`. The target was
|ΔlogLik| ≤ 1e-6 and relative estimate difference ≤ 1e-5. The largest
observed values are 1.4e-10 and 4.0e-10.

## How the two outputs line up

- **Order.** Julia `resd` = [ordinary bars in formula order..., marker].
  Native `sdpars$mu` uses the same order, for example `(1 | h)` then
  `phylo(1 | sp)`.
- **Names.** Julia uses `h` for `(1 | h)` and `h:x` for `(0 + x | h)`. The
  marker is named by its bare group, as in the single-structured routes. When
  an ordinary intercept shares the marker's group, as in `(1 | sp) +
  phylo(1 | sp)`, the ordinary term is named `sp_iid`.
- **Scale.** The marker SD is on the tree **correlation** scale, the same
  scale as native `ape::vcv(tree, corr = TRUE)`, so no tree-height rescaling
  is needed. This differs from the marker-only sparse phylo route, which
  reports raw branch-length scale and needs the R side's `sqrt(tree height)`
  conversion. The `rcoal` trees used here are not unit height, and the values
  match native without any conversion.

## Not covered (refused with `ArgumentError`, never dropped)

- `method = :REML`: native fits it, but Julia's REML validator still refuses
  this shape.
- A correlated `(1 + x | h)` alongside a marker.
- Range-estimated Julia `spatial(1 | site)` with `coords` (the bridge sends
  `relmat` + `K`, which is covered).
- `meta_V()`, `penalty`, and `algorithm = :em/:sparse/:sparse_lbfgs` with this
  shape.
- Two structured markers plus a bar. This was already refused.
- Random effects on `sigma`. These were already refused.
- Missing responses. This was already refused.
- Non-Gaussian families. They are outside this change; the R gate still
  refuses them.
- Scale. The route builds dense `n x n` matrices, like
  `_fit_two_structured_gaussian`. It is suited to thousands of rows, not tens
  of thousands.
- Standard errors and intervals. They come from the ForwardDiff Hessian and
  were not compared with native here; only point estimates, df, and logLik
  were compared.
