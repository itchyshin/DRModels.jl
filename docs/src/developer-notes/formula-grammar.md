# Formula grammar

!!! note "Status — Implemented (Gaussian surface)"
    Mirrors drmTMB's [Formula grammar](https://itchyshin.github.io/drmTMB/articles/formula-grammar.html). The grammar below — both `bf` shapes, the structured markers, and the reserved-syntax rejections — is **live in DRModels.jl today**. The Julia column is authoritative (taken from the source); for the R↔Julia spelling map see the [Rosetta — R ↔ Julia](@ref) page and the [R ↔ Julia bridge](@ref).

A DRModels.jl model is **one linear-predictor formula per distributional parameter**,
bundled by [`bf`](@ref) (alias `drm_formula`) and handed to [`drm`](@ref). This
mirrors `brms` / drmTMB: each formula's left-hand side names the parameter it
predicts. `bf` has two shapes — a **univariate** positional form and a
**bivariate** keyword form.

## Univariate form

```julia
bf(y ~ x1 + x2, sigma ~ x1)
```

- The **first** formula `y ~ …` sets the **response** column and the **mean `μ`**
  predictor. The mean is never written as a separate `mu ~ …` — μ *is* the
  response formula.
- Each **later** formula `param ~ …` sets that distributional parameter's
  predictor. The accepted secondary parameters are:

  | LHS name | Parameter | Link | Notes |
  |---|---|---|---|
  | `sigma` | scale σ | log | defaults to `~ 1` when omitted; **never `tau`** |
  | `sd(g)` | random-effect SD $\sigma_{b,k}$ | log | location–scale–scale on iid `(1 \| g)` RE |
  | `sd(sp, phylogenetic)` | phylogenetic SD $\sigma_{a,s}$ | log | location–scale–scale on `phylo(1 \| sp)` |
  | `nu`   | shape/df (e.g. Student-t) | — | family-dependent |
  | `zi`   | zero-inflation probability | — | |
  | `hu`   | hurdle probability | — | |
  | `zoi`  | zero-one-inflation | — | |
  | `coi`  | conditional-one-inflation | — | |

  Whether a given family actually *uses* a parameter (e.g. Gaussian has no `nu`)
  is a separate, family-level question resolved inside [`drm`](@ref).

`sigma` is special: if you omit it, `bf` appends `sigma => 1` automatically, so
`bf(y ~ x)` is `bf(y ~ x, sigma ~ 1)`.

### Two-column response

Families that take a count-of-trials response (binomial, beta-binomial) use
`cbind` on the left of the **mean** formula:

```julia
bf(cbind(successes, failures) ~ x, sigma ~ 1)
```

### The implicit intercept

DRModels.jl follows R's rule: `y ~ x` means `y ~ 1 + x` (an intercept is added). Drop
it explicitly with `0 + x` (or `-1`):

```julia
bf(y ~ x)        # ⇒ 1 + x   (intercept included)
bf(y ~ 0 + x)    # no intercept
```

This applies to every parameter's formula, mean and scale alike.

## Random effects and structured markers

Random-effect terms use the familiar `lme4` / `brms` spelling inside any
parameter's formula:

| Term | Meaning |
|---|---|
| `(1 \| g)` | random intercept by grouping factor `g` |
| `(1 + x \| g)` | correlated random intercept + slope |
| `(0 + x \| g)` | independent random slope |
| `(1 \| g) + (1 \| h)` | crossed random intercepts |

Wrapping a random-effect term in a **structured marker** gives it a known
correlation or sampling-variance structure:

```julia
bf(y ~ x + phylo(1 | species))        # phylogenetic correlation from a tree
bf(y ~ x + spatial(1 | site))         # spatial correlation from coords
bf(y ~ x + animal(1 | id))            # pedigree relatedness A
bf(y ~ x + relmat(1 | id))            # an arbitrary known relatedness matrix K
bf(y ~ meta_V(v))                     # meta-analysis: known sampling variances v
```

The marker functions (`phylo`, `spatial`, `animal`, `relmat`, `meta_V`) are
identity stubs that are **intercepted during formula parsing** — the structure
matrix itself (`tree`, `coords`, `A`, `K`) is supplied as a keyword to
[`drm`](@ref). See the [Structured-effect markers](@ref) reference.

## Bivariate keyword form

A two-response Gaussian model with a predictor-dependent residual correlation is
written with keywords, mirroring drmTMB:

```julia
bf(mu1   = @formula(y1 ~ x),
   mu2   = @formula(y2 ~ x),
   sigma1 = @formula(sigma1 ~ x),
   sigma2 = @formula(sigma2 ~ x),
   rho12  = @formula(rho12 ~ x))
```

- `mu1` / `mu2` set the two responses and their mean predictors.
- `sigma1` / `sigma2` are the two log-scales (log link); they default to `~ 1`.
- `rho12` is the **residual** correlation, on a `tanh` link (its coefficients act
  on `atanh ρ12`, keeping `ρ12 ∈ (-1, 1)`); it defaults to `~ 1`.

For the σ and ρ formulas, the left-hand side is a **placeholder** that must be the
parameter's own name (`sigma1 ~ …`, `rho12 ~ …`); only the right-hand side is
used. Group-level correlations (phylo / spatial / study) are *named covariance
summaries*, not this residual `rho12`.

### Structured bivariate Gaussian cells

Two structured bivariate Gaussian routes are recognised by the current Julia
front end.

The q=2 route puts matching structured random intercepts on `mu1` and `mu2`:

```julia
bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
   mu2 = @formula(y2 ~ x + phylo(1 | species)),
   sigma1 = @formula(sigma1 ~ 1),
   sigma2 = @formula(sigma2 ~ 1),
   rho12 = @formula(rho12 ~ 1))
```

This q=2 cell is exact-Gaussian ML only. The two mean formulas must use the
same fixed-effect design and the same grouping variable. `sigma1`, `sigma2`,
and `rho12` are intercept-only in this route. Complete responses are required.
The formula route currently accepts matching `phylo`, `relmat`, or `animal`
markers. The spatial q=2 evidence is a fixed-covariance direct fixture; it is
not a range-estimating spatial formula route.

The q=4 route is the phylogenetic location-scale cell with matching
`phylo(1 | group)` markers on all four axes:

```julia
bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
   mu2 = @formula(y2 ~ x + phylo(1 | species)),
   sigma1 = @formula(sigma1 ~ phylo(1 | species)),
   sigma2 = @formula(sigma2 ~ phylo(1 | species)),
   rho12 = @formula(rho12 ~ 1))
```

Direct q2/q4 export payloads are point-fit evidence only. They do not promote
broad R-via-Julia bridge support, q2/q4 REML, AI-REML, interval reliability, or
coverage.

## Reserved-syntax rejections

To catch the mistakes drmTMB catches, `bf` rejects the following with clear,
parallel error messages rather than building a wrong model:

| You wrote | Rejected because |
|---|---|
| `sigma` mistyped as `tau` | the scale parameter is named `sigma`, never `tau` |
| `mu ~ …` as a separate formula | μ comes from the response formula, not a separate one |
| `mu1`/`mu2`/`sigma1`/`sigma2`/`rho12` in the **positional** form | these are two-response parameters — use the keyword form |
| an unknown name (e.g. `theta ~ …`) | not a valid distributional parameter (the message lists the valid names) |
| the same parameter twice | one formula per distributional parameter |
| a bivariate placeholder LHS that isn't its own name (e.g. `sigma1 = @formula(tau ~ x)`, or swapped `sigma1`/`sigma2`) | each keyword's placeholder LHS must match its parameter |
| `cbind(…)` response on `mu1`/`mu2` | the bivariate form takes one response per mean (`cbind` is univariate-only) |

These rejections are front-end only — they never reach the fitter. See the
parity tests in `test/test_bf_grammar.jl`.
