# Arc 2 receipt: `meta_V` with a random effect on the mean

**Question.** Does DRModels.jl fit the same model as drmTMB (`engine = "tmb"`)
for Gaussian meta-analysis with known sampling variances plus a random
intercept on the mean: `meta_V(V = v) + (1 | study)`, `+ phylo(1 | sp)`,
`+ relmat(1 | id)`, and `phylo + (1 | study)` together?

**Answer.** Yes, on every fixture below: same df, logLik within 2.2e-10, every
estimate within 2.1e-9 relative (bar: 1e-6 absolute and 1e-5 relative).

## The gap this closes (measured at DRModels.jl da8b3f871)

From drmTMB's Arc 1 sweep (`meta_v_with_random_effect` gate, n = 180, 36
studies x 5): `meta_V + (1 | study)` returned the meta_V-only fit (julia df 3,
-161.907843; tmb df 4, -126.690704); `meta_V + phylo(1 | sp)` returned a
different model (julia -165.427760; tmb -161.759060); `relmat` the same. Cause,
read from `src/gaussian_core.jl`: the router's structured branch (which has no
known-variance term) and its `meta_V` branch (which has no random effect) both
ran before any branch that handled the combination, so each silently dropped
half of the formula.

## The model

drmTMB (`src/drmTMB.cpp`, univariate Gaussian branch): given u,
`y_i ~ N(x_i'b + (Zu)_i, V_known_i + sigma_i^2)`, `u_k ~ N(0, s_k^2 C_k)`,
integrated by Laplace, which is exact for this Gaussian-linear model. The
marginal is `N(Xb, diag(v + sigma^2) + sum_k s_k^2 Z_k C_k Z_k')`. DRModels'
new `_fit_meta_gaussian_re` (`src/gaussian_meta.jl`) maximises that marginal by
whitened Woodbury (O(n K^2 + q^3) per evaluation).

## Results (`comparison.tsv`, all PASS)

| fixture | formula | df (tmb = julia) | logLik (tmb) | max abs dlogLik | max rel d estimate |
|---|---|---|---|---|---|
| study-a (36 x 5, seed 20260924) | `meta_V + (1 \| study)` | 4 | -157.304457490 | 1.3e-10 | 8.8e-10 |
| study-b (24 x 6, seed 7) | `meta_V + (1 \| study)` | 4 | -152.112465749 | 1.2e-10 | 2.0e-09 |
| study-b | `meta_V + (1 \| study)`, `sigma ~ x` | 5 | -150.006233057 | 9.4e-11 | 1.8e-09 |
| phylo-a (30 tips, height 1.60) | `meta_V + phylo(1 \| sp)` | 4 | -115.686078396 | 1.1e-10 | 7.3e-10 |
| phylo-b (24 tips, height 3) | `meta_V + phylo(1 \| sp)` | 4 | -106.876807580 | 8.1e-11 | 7.8e-10 |
| relmat-a (25 ids x 6) | `meta_V + relmat(1 \| id)` | 4 | -125.084525197 | 2.1e-10 | 5.5e-10 |
| phylo-study | `meta_V + phylo(1 \| sp) + (1 \| study)` | 5 | -137.113642803 | 1.0e-10 | 9.7e-10 |

Estimates compared: mu coefficients, sigma coefficients (log heterogeneity),
and each random-effect SD. A phylo SD is reported by DRModels on the raw
branch-length scale, the convention of its default phylo-mean route; it is
multiplied by sqrt(mean root-to-tip depth), the factor drmTMB's own R bridge
applies (`drm_julia_phylo_sd_scale()`), before comparing. phylo-b's height of 3
makes that conversion visible (factor 1.73).

## Through drmTMB's R bridge (`r-bridge-probe.R`, `r-bridge-probe.txt`)

drmTMB 709efbeb0 (origin/main, before the Arc 1 refusal) with
`DRMODELS_JL_PATH` at this branch. Without help every new shape aborts in
DRModels' label echo: "coef_labels is missing an entry for dpar resd" (and,
for two components, "supplies 1 names but the resd formula part has 2").
With the label builder patched in-session only (one `resd` label per random
component, ordinary bars first then structured markers, each in formula
order), every shape fits through `engine = "julia"` with the same df, logLik
(to 1e-9 as printed) and fixed effects as `engine = "tmb"`. Two reporting gaps
remain on the R side:

- An ordinary `(1 | study)` SD is filed as `mu.study`, where native says
  `mu.(1 | study)`: the known labelling gap of the plain `(1 | g)` route.
- With two components (`phylo-study`), the phylo SD is filed as `mu.sp` and
  NOT converted by sqrt(height): 0.6080248 raw vs 0.7687283 native
  (0.6080248 x sqrt(1.598465) = 0.76873). The single-component phylo shape
  is filed and converted correctly (`mu.phylo(1 | sp)` = 0.6758138 on both).

The neighbours `meta_V` alone and with `sigma ~ x` are unchanged and match.

## Not covered

Refused with an error, each asserted in `test/test_meta_random_effect.jl`:
REML; random slopes `(0 + x | g)` / `(1 + x | g)` with `meta_V`; a `sigma`
random effect with `meta_V`; `sd(g) ~ ...` with `meta_V`; two components on
one grouping column (`phylo(1 | sp) + (1 | sp)`); `algorithm = :sparse`.
Refused by code read, not by a test here: `spatial()` with `meta_V`,
`sparse = true`, a `penalty`, and missing responses (the pre-existing
missing-response guard). Not expressible: a dense (non-diagonal) V, since
`meta_V(v)` takes a column of variances. The marginal bootstrap refuses a `meta_V` fit with more than
one random field (it draws one field only); single-field bootstrap draws the
full marginal (tested), with a phylo field placed on tree leaves by name, so a
non-tip species order and a tree with tips absent from the data are both drawn
correctly (tested); if the simulator cannot be built (no `tree`), bootstrap
refuses rather than fall back to the conditional draw. No interval-coverage claim.

## Reproduce

```sh
DRMTMB_PATH=<drmTMB with built DLL> Rscript docs/dev-log/evidence/arc2-metav-random-effect/native-fit.R
JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. docs/dev-log/evidence/arc2-metav-random-effect/julia-fit.jl
```

Environment: R 4.6.0, TMB 1.9.21, drmTMB 709efbeb07902fa748b124e1c268dbb8ac4ae683;
Julia 1.13.0, DRModels.jl branch `claude/arc2-metav` from da8b3f871; BLAS 1 thread.
