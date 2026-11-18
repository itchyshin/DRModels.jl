# When group variation changes with a predictor

!!! note "Checked example"
    The supplied Gamma example was run on one synthetic data set by maximum likelihood. This checks the point estimates and the named prediction outputs; it does not qualify other model combinations or interval coverage. The [asset and runtime manifest](https://raw.githubusercontent.com/itchyshin/DRModels.jl/e8a666769fc436e73c7121a4c49b1d9c92866776/docs/src/assets/sd-regression/manifest.json) identifies the data and tested Julia source.

Does an environmental gradient change how much groups differ, as well as their average response? A model for the random-effect standard deviation (SD) can answer that question. This example uses a positive response, a Gamma distribution and a supplied matrix of relationships among groups. It accompanies the R article [When group variation changes with a predictor](https://itchyshin.github.io/drmTMB/articles/predictor-dependent-random-sd.html).

For a Gaussian response, the [location-scale-scale tutorial](location-scale-scale.md) also lets predictors change the residual SD. Here the three quantities have different roles: `mu` is the conditional response mean, `sigma` is a constant coefficient of variation for the Gamma response, and `sd(id, relmat)` controls the spread of the latent group intercept.

## The model and its scale

For observation `i` in group `g`, fit

```math
\begin{aligned}
Y_{ig}\mid b_g &\sim \operatorname{Gamma}\!\left(
  \text{shape}=1/\sigma^2,\;\text{rate}=1/(\sigma^2\mu_{ig})\right),\\
\log\mu_{ig} &= \beta_0+\beta_1 x_{ig}+b_g,\\
\mathbf b &\sim N(\mathbf0,\;D K D),\qquad
D=\operatorname{diag}(s_1,\ldots,s_G),\\
\log s_g &= \alpha_0+\alpha_1 z_g.
\end{aligned}
```

`z` must be constant within each group. A unit increase in `z` multiplies the SD multiplier `s_g` by `exp(alpha_1)`. The corresponding variance multiplier changes by `exp(2alpha_1)`.

Keep the supplied matrix on its original scale. The marginal latent-effect SD for group `g` is `s_g * sqrt(K[g,g])`. It equals `s_g` only when the diagonal entry is one. If two groups have different diagonal entries, their marginal SD ratio also includes the ratio of those square roots.

The formula `sigma ~ 1` estimates one constant Gamma coefficient of variation. It does not fix its value. Conditional on `b_g`, the response SD is `sigma * mu_ig`, so response spread can increase with the mean even when `sigma` is constant.

## Read the shared example

Download [the data](https://raw.githubusercontent.com/itchyshin/DRModels.jl/e8a666769fc436e73c7121a4c49b1d9c92866776/docs/src/assets/sd-regression/gamma-data.tsv), [the matrix](https://raw.githubusercontent.com/itchyshin/DRModels.jl/e8a666769fc436e73c7121a4c49b1d9c92866776/docs/src/assets/sd-regression/gamma-K.tsv), and [its labels](https://raw.githubusercontent.com/itchyshin/DRModels.jl/e8a666769fc436e73c7121a4c49b1d9c92866776/docs/src/assets/sd-regression/gamma-K-labels.tsv) into your working folder. The [asset manifest](https://raw.githubusercontent.com/itchyshin/DRModels.jl/e8a666769fc436e73c7121a4c49b1d9c92866776/docs/src/assets/sd-regression/manifest.json) records their hashes. `gamma-data.tsv` contains numeric `y`, `x` and `z` plus group label `id`; `gamma-K.tsv` contains a numeric square matrix without headers; and `gamma-K-labels.tsv` lists its row labels under an `id` header. The matrix has the same row and column labels. These are synthetic data, not observations from a field study.

These supplied files contain no quoted fields or embedded tabs. The small reader below uses Julia's standard functions and is intended for these example assets.

```julia
using DRModels, LinearAlgebra

lines = readlines("gamma-data.tsv")
header = split(first(lines), '\t')
raw = permutedims(reduce(hcat, split.(lines[2:end], '\t')))
column(name) = raw[:, only(findall(==(name), header))]
dat = (y = parse.(Float64, column("y")),
       x = parse.(Float64, column("x")),
       z = parse.(Float64, column("z")), id = String.(column("id")))

K_rows = [parse.(Float64, split(line, '\t'))
          for line in readlines("gamma-K.tsv")]
K_file = permutedims(reduce(hcat, K_rows))
k_labels = readlines("gamma-K-labels.tsv")[2:end]
groups = unique(dat.id)  # first appearance in the data
@assert length(unique(k_labels)) == length(k_labels)
@assert size(K_file) == (length(k_labels), length(k_labels))
order = [only(findall(==(id), k_labels)) for id in groups]
K = K_file[order, order]
@assert isposdef(Symmetric(K))
@assert all(K .== K')
```

DRModels pairs the first matrix row with the first group to appear in the data. Reordering by label before fitting prevents a shuffled data table from changing which groups are related. Do not rescale `K` to unit diagonal for this example.

## Fit by maximum likelihood

```julia
form = bf(@formula(y ~ x + relmat(1 | id)),
          @formula(sigma ~ 1),
          @formula(sd(id, relmat) ~ z))
fit = drm(form, DRModels.Gamma(); data = dat, K = K,
          method = :ML, marginal = :LA)

alpha = coef(fit, :sd)
(alpha = alpha, sd_multiplier_ratio = exp(alpha[2]),
 gamma_cv = exp(only(coef(fit, :sigma))))
```

The structured Gamma route uses a Laplace approximation to integrate over the latent group effects. This statement applies to this supplied-`K`, scalar-intercept model. The `:LA` option does not identify the same numerical integration rule on every family and random-effect layout. For example, the ordinary one-group Poisson SD route uses adaptive quadrature.

For this formula, `coef(fit, :sd)` contains the intercept and the slope for numeric `z`. The slope describes the SD multiplier on the latent log-mean scale. The supplied data were generated with `alpha_z = 0.4`, so the true one-unit multiplier ratio is 1.4918247. Direct Julia estimated a ratio of 1.0729704, with log likelihood -572.328865533. The independent Laplace calculation differs by about 2e-10, below the prespecified absolute tolerance of `1e-5`.

On these same data, the checked R native and R-to-Julia bridge fits estimated the same ratio to seven decimal places. Agreement among implementations and the independent likelihood calculation is a check on this fit. It does not establish that a single data set recovers the generating slope; here the estimated ratio is well below the true ratio.

## Predict the multiplier and marginal SD

```julia
parameters = predict_parameters(fit, dat)
s_observation = parameters[:sd]  # one value per observation
first_row = [findfirst(==(id), dat.id)::Int for id in groups]
s_group = s_observation[first_row]
marginal_sd = s_group .* sqrt.(diag(K))

(group = groups, sd_multiplier = s_group,
 marginal_latent_sd = marginal_sd)
```

Both outputs describe latent group variation. Neither is the Gamma response SD. Repeated rows for the same group have the same SD multiplier because `z` is constant within that group.

To compare two specified predictor values, supply the grouping column as well as the model predictors. Each group in a prediction table must have one SD-design row. `predict_parameters` evaluates the fitted SD regression; it does not estimate a new latent group effect.

## Know which response prediction you requested

```julia
mu_at_zero = predict(fit, dat)
mu_link_at_zero = predict(fit, dat; type = :link)
```

These direct Julia predictions set the latent effects to zero: they return `exp(X*beta)` and `X*beta`, respectively. Direct Julia `fitted(fit; re = :zero)` has the same response-scale convention for this route; the default `fitted(fit)` adds the fitted random-effect modes, as drmTMB's `fitted()` does, where the route keeps them. Setting random effects to zero does not integrate over their distribution, so this is not a marginal response mean.

The R bridge has a different stored-row convention when the Julia fitter has retained a valid structured conditional field. Its stored `fitted()` and `predict()` values include the fitted latent field on the checked structured route. Supplying `newdata` through the bridge still sets random effects to zero. Label these quantities explicitly when comparing R and Julia output. Compare like quantities before interpreting a numerical difference as a fitting difference.

## Limits of this example

This example covers maximum likelihood for one supplied relatedness matrix, one scalar mean random intercept, a fixed-predictor SD formula and one estimated constant Gamma `sigma`. Use complete, strictly positive responses. The supplied matrix must cover the observed groups, and the group-level SD design must have independent columns. If a predictor varies within a group, revisit the scientific model rather than averaging it solely to bypass the check.

A successful fit and point prediction do not establish calibrated interval coverage. Interval calibration and REML treatment remain outside this example. This page makes no broad claim about other distribution/provider combinations.

For the Gaussian case, continue with [location-scale-scale models](location-scale-scale.md). For the constant-variance Gaussian supplied-matrix model, see [known-matrix relatedness](relmat-known-matrices.md); that simpler model uses a different fitting method.
