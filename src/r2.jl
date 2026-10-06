# r2.jl — R² for the ONE case where it means exactly one thing.
#
# WHY THIS IS NARROW, and why the refusals are the point.
#
# R² answers "what fraction of the variance is explained". In a distributional
# regression the total variance is ITSELF MODELLED — it changes with the
# covariates — so there is no single denominator to divide by. Variance at the
# mean covariate? Averaged over the sample? With σ held at its intercept? Each
# gives a different number, and a reader arriving from `lm()` will assume the
# `lm()` meaning and be wrong.
#
# So this function computes R² only where the question has one answer — a
# Gaussian mean model with CONSTANT σ and no random effects — and REFUSES,
# loudly and by name, everywhere else. A refusal that says which assumption
# failed teaches more than a number that quietly picked a denominator.
#
# Deliberately NOT provided: a general `r2`. gamlss and glmmTMB export none
# either; brms ships `bayes_R2` and is explicit that it is a different quantity.

"""
    r2_constant_sigma(fit::DrmFit) -> Float64

Coefficient of determination for a Gaussian mean model with **constant σ**:

```
R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

This is the ordinary `lm()` R², and on a constant-σ Gaussian fit it equals what
`lm()` reports for the same mean model.

It **refuses**, rather than returning a number, whenever that formula would be
ambiguous or misleading:

* **a modelled σ** (any non-intercept term in the `sigma` formula) — the total
  variance then depends on the covariates, so "the" denominator does not exist;
* **random effects** — the quantity wanted there is a *marginal* or *conditional*
  R² (Nakagawa & Schielzeth), which are two different numbers, neither of them
  this one;
* **a non-Gaussian family**, or a **bivariate** fit — a different definition
  applies, not this one;
* **a mean model with no intercept** — `Σ(y - ȳ)²` is then not the baseline the
  model is being compared against.

# Examples
```julia
fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)
r2_constant_sigma(fit)      # e.g. 0.397

bad = drm(bf(@formula(y ~ x), @formula(sigma ~ sex)), Gaussian(); data = d)
r2_constant_sigma(bad)      # ArgumentError naming the modelled σ
```

See also [`coeftable`](@ref), [`summary`](@ref).
"""
function r2_constant_sigma(fit::DrmFit)
    blocks = Dict(p => r for (p, r) in fit.blocks)
    names  = Dict(p => v for (p, v) in fit.coefnames)

    # --- refuse: bivariate --------------------------------------------------
    if haskey(blocks, :mu1) || haskey(blocks, :mu2)
        throw(ArgumentError(
            "r2_constant_sigma is defined for a univariate Gaussian mean model; " *
            "this is a bivariate fit. There is no single `y` to decompose — report " *
            "per-response fits, or a quantity chosen for the bivariate case."))
    end

    # --- refuse: non-Gaussian ----------------------------------------------
    if !(fit.family isa Gaussian)
        throw(ArgumentError(
            "r2_constant_sigma is defined for `Gaussian()` only; this fit uses " *
            "$(typeof(fit.family)). For non-Gaussian families the variance is a " *
            "function of the mean, so 1 - RSS/TSS is not the fraction of variance " *
            "explained. Nothing is computed rather than reporting a number that " *
            "does not answer the question."))
    end

    # --- refuse: random effects --------------------------------------------
    if fit.ranef !== nothing
        throw(ArgumentError(
            "r2_constant_sigma refuses a fit with random effects. What is wanted " *
            "there is a MARGINAL R² (fixed effects only) or a CONDITIONAL R² " *
            "(fixed + random) in the sense of Nakagawa & Schielzeth — two different " *
            "numbers, neither equal to 1 - RSS/TSS. Picking one silently would " *
            "misreport whichever the reader had in mind."))
    end

    # --- refuse: modelled sigma (the case this function exists to catch) ----
    for p in (:sigma, :sigma1, :sigma2)
        haskey(blocks, p) || continue
        nms = get(names, p, String[])
        extra = filter(n -> n != "(Intercept)", nms)
        if !isempty(extra)
            throw(ArgumentError(
                "r2_constant_sigma refuses a MODELLED σ: the `$(p)` block carries " *
                "$(length(extra)) non-intercept term(s) ($(join(extra, ", "))). " *
                "The total variance then changes with the covariates, so R² has no " *
                "single denominator — variance at the mean covariate, averaged over " *
                "the sample, and at σ's intercept all give different answers. Fit " *
                "`sigma ~ 1` if you want the ordinary R²."))
        end
    end

    # --- refuse: no intercept in the mean model -----------------------------
    munames = get(names, :mu, String[])
    if !("(Intercept)" in munames)
        throw(ArgumentError(
            "r2_constant_sigma refuses a mean model with no intercept: Σ(y - ȳ)² " *
            "is not the baseline such a model is being compared against, and the " *
            "resulting number is not comparable with an intercept model's."))
    end

    # --- the one well-defined case -----------------------------------------
    haskey(fit.obs, :mu) && haskey(fit.means, :mu) || throw(ArgumentError(
        "r2_constant_sigma needs the observed response and fitted means; this fit " *
        "carries neither under `:mu`."))
    y = fit.obs[:mu]
    ŷ = fit.means[:mu]
    length(y) == length(ŷ) || throw(ArgumentError(
        "r2_constant_sigma: response and fitted values differ in length " *
        "($(length(y)) vs $(length(ŷ)))."))

    ss_res = sum(abs2, y .- ŷ)
    ss_tot = sum(abs2, y .- (sum(y) / length(y)))
    ss_tot > 0 || throw(ArgumentError(
        "r2_constant_sigma: the response has zero variance, so R² is undefined."))
    return 1 - ss_res / ss_tot
end
