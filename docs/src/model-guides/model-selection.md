# Model selection with AIC and BIC

!!! note "Status — Stable"
    Mirrors drmTMB's [Model selection with AIC and BIC](https://itchyshin.github.io/drmTMB/articles/model-selection.html).
    **In DRModels.jl today:** information-criterion comparison (`aic`, `bic`,
    [`aicc`](@ref)) and the nested likelihood-ratio test ([`lrtest`](@ref) /
    [`anova`](@ref)) over fitted `drm` models, plus [`check_drm`](@ref) for the
    convergence / boundary diagnostics that belong beside any criterion table.

Model selection comes *after* candidate models have been chosen, fitted, and
checked. AIC and BIC can rank fitted distributional-regression models, but they
do not decide whether a candidate set is biologically sensible, nor whether a
weak fit should be interpreted at all.

Both criteria use the fitted log-likelihood and the number of estimated
parameters:

```text
AIC = -2·loglik + 2·k
BIC = -2·loglik + log(n)·k
```

`k` is the model degrees of freedom ([`dof`](@ref)) and `n` is the number of
observations ([`nobs`](@ref)). BIC penalises extra parameters more strongly than
AIC once `n > exp(2) ≈ 7.4`. Compare models **only** when they were fit by the
same method (ML — DRModels.jl's default) to the same response, on the same rows. Do
not compare a model fit to raw counts with one fit to transformed counts, or two
fits that silently dropped different rows.

A small helper builds a criterion table across a named set of fits. It reads
only public accessors, so it works for every family:

```@example modsel
using DRModels, Random

function criterion_table(models::Pair{Symbol,<:Any}...)
    aics = [aic(m) for (_, m) in models]
    bics = [bic(m) for (_, m) in models]
    (
        model      = [string(name) for (name, _) in models],
        AIC        = aics,
        BIC        = bics,
        converged  = [is_converged(m) for (_, m) in models],
        ΔAIC       = aics .- minimum(aics),
        ΔBIC       = bics .- minimum(bics),
    )
end
nothing # hide
```

## Does the residual scale need a predictor?

Model selection is not only family selection — it can ask whether a
*distributional parameter* earns a predictor. Here the mean changes with `x`,
and the residual scale changes with `x` too. We compare a constant-scale model
(`sigma ~ 1`) against a moving-scale model (`sigma ~ x`):

```@example modsel
Random.seed!(2403)
n = 220
x = randn(n)
logσ = -0.45 .+ 0.55 .* x          # the spread grows with x
y = 0.3 .+ 0.55 .* x .+ exp.(logσ) .* randn(n)
scale_dat = (; y, x)

fit_sigma_constant = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = scale_dat)
fit_sigma_x        = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = scale_dat)

criterion_table(Symbol("sigma ~ 1") => fit_sigma_constant,
                Symbol("sigma ~ x") => fit_sigma_x)
```

The lower-AIC/BIC model wins *inside this two-model set*. Because the models are
nested (the constant-scale model is the moving-scale model with the σ-slope set
to zero), you can also test the extra parameter directly with the
likelihood-ratio test:

```@example modsel
lrtest(fit_sigma_constant, fit_sigma_x)   # (; statistic, dof, pvalue)
```

A small p-value supports the moving-scale term within these nested, checked
candidate models; it is not proof of a biological mechanism. Interpret the selected
σ-coefficient on the **ratio** scale — a one-unit increase in `x` multiplies the
residual SD by `exp(slope)`:

```@example modsel
exp(coef(fit_sigma_x, :sigma)[2])         # residual-SD ratio per unit x
```

## A tail assumption: Gaussian vs Student-t

A robust [`Student`](@ref) model is useful when the process can produce
unusually large residuals. Here the data are generated with heavy-tailed
(t with 4 df) residuals, then fit with a Gaussian model and a Student-t model
that share the same `mu` and `sigma` formulas:

```@example modsel
using Distributions: TDist
Random.seed!(2401)
n = 220
x = randn(n)
tail_dat = (; x, y = 0.2 .+ 0.7 .* x .+ exp(-0.25) .* rand(TDist(4.0), n))

fit_tail_gaussian = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)),
                        Gaussian(); data = tail_dat)
fit_tail_student  = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
                        Student(); data = tail_dat)

criterion_table(:Gaussian => fit_tail_gaussian, Symbol("Student-t") => fit_tail_student)
```

The lower-criterion model is the better fit in this candidate set — but that is
not a licence to ignore diagnostics. Check whether the fitted degrees of freedom
`ν` is so large that the Student-t is effectively Gaussian (`ν` uses the logm2
link `ν = 2 + exp(η)`, so read it back with `2 + exp(...)`):

```@example modsel
2 + exp(coef(fit_tail_student, :nu)[1])   # small ν ⇒ robustness is earning its keep
```

## Structural zeros: NB2 vs zero-inflated NB2

For count responses, the extra parameter in a zero-inflated model has a specific
meaning: the probability of a separate structural-zero process. This example
generates NB2 counts with structural zeros and compares an ordinary NB2 fit
against a zero-inflated one (`zi ~ 1`):

```@example modsel
using Distributions: NegativeBinomial
Random.seed!(2402)
n = 260
x = randn(n)
μ  = exp.(log(2.3) .+ 0.5 .* x)
ϕ  = 1 / 0.65^2                           # NB2 size from the dispersion
zi = 1 / (1 + exp(0.8))                   # structural-zero probability ≈ logistic(-0.8)
count = [rand(NegativeBinomial(ϕ, ϕ / (ϕ + μ[i]))) for i in 1:n]
count[rand(n) .< zi] .= 0
count_dat = (; count = float.(count), x)

fit_nb2   = drm(bf(@formula(count ~ x), @formula(sigma ~ 1)),
                NegBinomial2(); data = count_dat)
fit_zinb2 = drm(bf(@formula(count ~ x), @formula(sigma ~ 1), @formula(zi ~ 1)),
                NegBinomial2(); data = count_dat)

criterion_table(:NB2 => fit_nb2, :ZINB2 => fit_zinb2)
```

If the criterion difference is small, keep both explanations in view: an NB2 can
absorb some extra zeros by inflating its overdispersion, while a ZINB2 separates
structural zeros from count variation. Use `zi` only when a structural-zero
process is plausible for the response and sampling design.

## Diagnostics belong beside the table

A criterion is a number; it cannot tell you whether the winning fit is *stable*.
[`check_drm`](@ref) returns the convergence, gradient, and boundary status that
must travel with any selection decision:

```@example modsel
check_drm(fit_zinb2)
```

Exclude errored fits from selection, and treat a non-converged or boundary fit
as a diagnostic finding, not as an ordinary winner.

## Practical checklist

When comparing fitted `drm` models:

1. Define the candidate set **before** looking at AIC/BIC.
2. Fit every candidate by the same method (ML), to the same response and rows.
3. Run [`check_drm`](@ref) and keep convergence and boundary status beside the
   criterion table.
4. Exclude errored fits; treat non-converged or boundary fits as diagnostic
   findings, not winners.
5. Report AIC and BIC **differences** (`ΔAIC`, `ΔBIC`), not only the winner.
6. Interpret the selected model's `mu`, `sigma`, `nu`, or `zi` parameters in
   scientific units or ratios.

When AIC and BIC disagree, describe the tradeoff rather than declaring one
criterion universally correct. AIC is more willing to keep an extra parameter
that improves fit; BIC asks for stronger evidence as `n` grows. In applied work,
that disagreement is usually a cue to inspect predictions, residuals, and the
*meaning* of the extra distributional parameter.

!!! tip "Small samples: prefer AICc"
    When `n / k` is small (a common rule of thumb is `n / k < 40`), use
    [`aicc`](@ref) — the second-order correction `AICc = AIC + 2k(k+1)/(n−k−1)`,
    which is always ≥ AIC and converges to it as `n → ∞`.

## See also

- [Choosing a model & checking it](model-workflow.md) — the fit-then-check loop.
- [Did it converge?](convergence.md) — reading [`check_drm`](@ref) in full.
- [Prediction, residuals & model comparison](../diagnostics-and-validation/prediction-and-postfit.md) — the post-fit accessor tour.
