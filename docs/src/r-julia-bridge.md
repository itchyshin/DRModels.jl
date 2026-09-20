# R ↔ Julia bridge

!!! note "Status — experimental optional bridge"
    DRModels.jl exposes `drm_bridge()`, a marshalling-friendly entry point used by
    the optional `drmTMB(formula, ..., engine = "julia")` backend for supported
    models. The companion R glue lives in the **drmTMB R repository** via
    [JuliaCall](https://github.com/JuliaInterop/JuliaCall); the default
    `engine = "tmb"` does not require Julia.

## Start with your current workflow

Stay with native `drmTMB` when you want its default, established R/TMB route.
Choose the bridge only when your model is within its admitted scope and you want
to try the Julia backend without rewriting your formula or data workflow.

If you want to keep writing R, follow drmTMB's
[Julia-engine setup and ordinary-regression example](https://itchyshin.github.io/drmTMB/articles/julia-engine.html).
It installs Julia and JuliaCall, points R to a local DRModels.jl checkout, and
then fits the same R formula with `engine = "julia"`. Keep `engine = "tmb"`
until that optional setup is complete.

If you want to write Julia directly, begin with
[Getting started](getting-started.md), then use the [Rosetta](rosetta.md) to
translate a model you already understand in R.

## When native R is the better route

Keep `engine = "tmb"` when your study needs an established R workflow that is
outside this bridge. In particular, use drmTMB's
[Coordinate-spatial structured effects](https://itchyshin.github.io/drmTMB/articles/spatial-models.html)
for its distributional-regression spatial route; use gllvmTMB's
[Multivariate spatial models with an SPDE mesh](https://itchyshin.github.io/gllvmTMB/articles/spatial-models.html),
[Temporal covariance for repeated multivariate measurements](https://itchyshin.github.io/gllvmTMB/articles/temporal-ar1.html),
or [What do repeated survey visits add to an integrated model?](https://itchyshin.github.io/gllvmTMB/articles/integrated-repeated-visits.html)
when those match your question. These native R spatial, temporal, and
repeated-visit workflows are not admitted to `engine = "julia"`; a successful
Julia setup does not extend the bridge to them.

The bridge is deliberately limited to the R workflows named below. The
documented examples agree with matching R fits, but that is not a general
speed, small-sample, or interval-reliability claim. For translating R syntax
to Julia by hand, see the [Rosetta page](rosetta.md).

## The idea

Two ways to use DRModels.jl from R, in increasing integration:

1. **Translate by hand** — rewrite the model in Julia using `drm` / `bf`. The
   [Rosetta](rosetta.md) phrasebook is the lookup table. Available today.
2. **`engine = "julia"`** — keep writing ordinary `drmTMB(...)` R code; drmTMB
   marshals the formula and data across JuliaCall, calls DRModels.jl to fit, and
   returns a result object shaped like a native drmTMB fit. Supported for
   Gaussian one-response and two-response models, the first Gaussian
   `phylo(1 | species)` mean bridge with constant `sigma`, location–scale–scale
   `sd(group)` / `sd(species, level = "phylogenetic")` models (ML, REML, sparse
   scaling, and missing response inclusion), narrow complete-response q=2
   structured Gaussian examples, and eleven checked coefficient-and-scale
   model types. The bridge remains optional; R-side setup lives in drmTMB.

### Trees with polytomies

A polytomy is an internal node with more than two immediate children. The bridge
accepts these nodes without resolving them into invented binary branches.
The tree must still have positive branch lengths, nonempty unique tip labels, and
at least two children at every internal node. The R bridge currently requires an
ultrametric tree for its correlation-scale convention. Zero-length branches
and unary nodes are not currently supported through the R bridge.

The bridge preserves tip labels containing spaces, punctuation,
Unicode and apostrophes. Keep the same labels in your data; do not replace spaces
with underscores. The serializer quotes labels where needed, and Julia decodes
them without changing their spelling. In direct Newick input, use single quotes
around such labels and double an apostrophe inside a quoted label:

```@example quoted_tree_labels
using DRModels
named_tree = augmented_phy("('Mola mola':1,'O''Brien':1,A_B:1);")
@assert named_tree.leaf_names == ["Mola mola", "O'Brien", "A_B"]
named_tree.leaf_names
```

Quoted labels preserve whitespace literally, including leading/trailing spaces.
This concerns tip identity; internal-node labels are parsed but not retained.

Direct Julia keeps the supplied Brownian branch-length scale and can represent
unequal tip depths:

```@example polytomy_tree
using DRModels
phy = augmented_phy("((A:1,B:2,C:3):4,D:5,E:6);")
@assert phy.n_leaves == 5 && phy.n_total == 7
@assert phylo_tree_height(phy) == 7
# Small diagnostic only: a dense tip covariance is unsuitable for large trees.
sigma_phy_dense(phy)
```

For an ultrametric tree of height `h`, Julia's raw phylogenetic SD multiplied by
`sqrt(h)` is on the correlation scale used by the R bridge. One height cannot
standardize a tree whose tip depths differ. Accepting a topology does not itself
verify every response family, profile interval or bootstrap workflow.

### One modelled missing predictor — development admission

!!! warning "Experimental"
    Exported for evaluation. API and numerics may
    change. It is not an established replacement for the native R route.

The R bridge also has a deliberately narrow development route for one modelled
missing predictor. It accepts a Gaussian identity-link response, exactly one
bare additive `mi(x)` term in `mu`, complete fixed-effect exogenous designs,
and a Gaussian, Bernoulli, ordinal or categorical fixed-effect predictor model. The direct
Julia frontend and `drmTMB(..., engine = "julia")` use the same prepared joint
likelihood; observed `x` values remain observed and missing `x` values are
integrated rather than filled before fitting.

Use `impute = list(x = x ~ z)` for a Gaussian predictor, or
`impute = list(x = impute_model(x ~ z, family = binomial()))` for a binary
predictor, with either `missing = miss_control(response = "drop", predictor =
"model")` or `miss_control(response = "include", predictor = "model")`. The
bridge response-drop path removes missing-response rows before preparation.
That is a documented preprocessing choice, not native response-policy parity.

This admission is not a general missing-data bridge. It rejects other response
families, interactions or nesting involving `mi()`,
random or structured effects, offsets, non-default controls, likelihood weights,
and REML. `summary()` and Wald `confint()` are available only when the returned
covariance is usable; profile and bootstrap intervals explicitly error. The
Gaussian predictor-SD interval is a natural-scale delta-Wald interval, may cross
zero, and is not claimed to match native intervals or to have established
coverage.

Two public bridge-adapter cases pass. Training prediction and binary
`newdata` handling have been repaired and checked independently. Full numerical
parity remains open: small differences in native optimizer stopping affect
coefficients and predictions beyond the declared tolerance. This route makes
no full native-parity, speed, or interval-coverage claim. A separate development
route also admits two independent Gaussian predictors; this does not admit
arbitrary combinations of missing-predictor families.

For an ordered predictor, use an ordered R factor and
`impute_model(x ~ z, family = cumulative_logit())`; for a nominal predictor,
use a factor and `impute_model(x ~ z, family = categorical())`. Both finite-state
routes require at least three observed levels. The ordered predictor model
removes its intercept because its cutpoints already supply location parameters.
The response mean still follows its own formula's intercept convention.

Finite-state predictions average the fitted mean over posterior states.
`imputed()` returns expected category scores and conditional score SDs for an
ordered predictor, or the first modal category code for a nominal predictor.
Nominal metric standard errors are unavailable. These are conditional summaries,
not multiple-imputation draws. Ordered-predictor cutpoints are retained in
`fit$missing_data$predictors$x$cutpoints`; ordinary R `coef()` and `vcov()` exclude
them. The bridge retains all raw covariance coordinates internally.

The two retained finite-state bridge cases pass transport and public-operation
checks, including new-data predictions. Native numerical parity remains open
at the unchanged `4e-6` tolerance; these checks establish neither faster warm
workflows nor the full native missing-data interface.

## The DRModels.jl-side contract

For the bridge to work, DRModels.jl exposes a stable, marshalling-friendly surface:

- **Formula** — the R `bf(mu = y ~ x, sigma = ~ x, ...)` is mapped to DRModels.jl's
  `bf(...)`; the [Rosetta formula grammar](rosetta.md#formula-grammar) gives
  the reader-facing spelling map;
- **Data** — an R `data.frame` crosses as a column table (`NamedTuple` /
  `DataFrame`) keyed by the same column names;
- **Result** — `drm_bridge()` returns coefficient names and values, an available
  covariance matrix, likelihood summaries, convergence information, fitted
  values, residuals, and fitted scale. Some advanced outputs are available only
  for the particular model types listed above; they are not a general guarantee
  of interval or coverage performance.

For the Gaussian phylogenetic mean cell, the current `algorithm = :auto` route
uses an all-node sparse L-BFGS fitter. That route
profiles the mean coefficients by sparse GLS, uses exact Takahashi trace
gradients for the residual and phylogenetic standard deviations, and returns a
finite mean-coefficient covariance block. Scale and variance-component
covariance are still left unset for the R bridge, so profile/bootstrap work
remains the next inference slice.

## R formula constructs through `engine = "julia"`

R's formula mini-language is not Julia's. `@formula` cannot evaluate `poly(x, 3)` or
`factor(g)` the way an R user means them, so the bridge **rewrites** each construct into
materialised columns or an expanded term list *before* handing the formula to
`@formula`. Every construct below is either implemented with a matched R-parity
reference on byte-identical data, or rejected for a measured reason.

| construct | status |
|---|---|
| `I(expr)` | supported, over a safe `+ - * / ^` grammar only — never arbitrary code |
| `scale(x)` | supported; centres and scales by the sample mean/SD (R's default, `n-1` denominator) |
| `factor(g)` | supported; levels ordered by `sort(unique(...))` on the original values, matching R's `contr.treatment` baseline |
| `(...)^k` | supported for a literal positive integer `k` over a `+`-only expression |
| `- term` | supported, including general term removal |
| `poly(x, k)` | supported — R's **orthogonal** basis (`raw = FALSE`, the default), expanding to `k` columns |
| `poly(x, k, raw = TRUE)` | rejected — write the powers explicitly with `I(x^k)` |
| `poly()` inside `(...)^k` | **rejected on measured evidence**, see below |
| `poly()` inside a scalar function | rejected, e.g. `log1p(poly(x, 2))` |
| `poly(x, y, degree)`, explicit `coefs =` | rejected — precompute in R and pass the columns |

### Two rejections worth explaining

`poly()` is the only construct that rewrites to a **group** of terms, and a group does not
compose everywhere a single column does.

**Inside `(...)^k`.** R treats `poly(x, 2)` as **one term**, so `(x + poly(x, 2))^2` crosses
two terms and never forms `poly1:poly2`. Measured against `model.matrix()` on both sides:
R produces **6** model-matrix columns, the flattened rewrite produces **7** — the extra one
being exactly that interaction. Rejected rather than special-cased, because keeping the group
intact through the power algebra needs a term-grouping concept this rewrite does not have.

**Inside a scalar function.** R maps the function elementwise over poly's `k`-column matrix,
giving `k` columns; the rewrite would map it over their **sum**, giving one. Silent, and of
exactly the shape a bridge exists to prevent.

Where poly *does* compose, it was checked rather than assumed — `x * poly(x, 2)`,
`z : poly(x, 2)`, `poly(x, 2) + z` and `x + z - poly(x, 2)` all match R's model-matrix width
exactly.

### Materialised columns and `newdata`

`I()`, `scale()`, `factor()` and `poly()` are translated before fitting. The
bridge retains the corresponding formula labels and an explicit
mapping to the fitted coordinates. Use the returned public coefficient name,
such as `I(x^2)`, in `parm = "fixef:mu:I(x^2)"`; do not guess a temporary column
number. Existing data columns are never replaced by a generated column.

The companion R adapter uses the original training terms for supported
fixed-effect `predict(..., newdata = ...)` calls. In particular, `scale()` keeps
the training mean and standard deviation, and `poly()` keeps its training
orthogonal basis. Rebuilding either basis on the new rows would change the
prediction. This R adapter behavior does not imply that a direct Julia
`DrmFit` can reconstruct arbitrary materialised columns for new data.

These changes require the matching development versions of both packages.
Legacy bridge objects without label metadata keep their existing names.
Ambiguous or incomplete label metadata is rejected. This is a coefficient
identity contract; it does not establish interval coverage or large-tree
profile performance.

```@example bridge_coefficient_labels
using DRModels
x_labels = collect(range(-1.5, 1.5; length = 48))
y_labels = 0.2 .+ 0.4 .* x_labels .- 0.1 .* x_labels.^2 .+
           0.15 .* sin.(collect(1:48))
label_fit = drm_bridge(formula = "y ~ x + I(x^2); sigma ~ 1",
    family = "gaussian", data = (; y = y_labels, x = x_labels))
@assert "mu_I(x^2)" in label_fit["coef_names"]
@assert label_fit["coef_names"] == label_fit["vcov_names"]
label_fit["coef_names"]
```

## What has been compared

The admitted Gaussian and selected fixed-effect model types have been compared
with known R examples. That comparison helps catch translation mistakes. It
does not establish that the two packages agree for every family, data shape, or
scientific estimand.

## Open design questions

The following boundaries remain:

- **Broader phylogenetic, pedigree, and relatedness models** — the first
  Gaussian phylogenetic mean model is supported. Multiple structured terms,
  slopes, and non-Gaussian phylogenetic models need separate work.
- **Exact R-object matching** — a Julia-engine result is designed to be useful
  in an R workflow, but it is not field-for-field identical to every native
  drmTMB result.
- **More formula forms** — each new formula feature needs its own comparison
  before it can be offered through the bridge.
- **Broad performance comparisons** — do not infer a speed advantage from the
  supported examples; performance depends on model size and structure.

Use the bridge for the documented Gaussian one-response and two-response
workflows, including the first supported Gaussian phylogenetic mean model. Use
hand-translation via the Rosetta phrasebook or native `drmTMB` for remaining
families and unsupported formula features.
