# comparison.jl — model-comparison + small accessor parity (drmTMB / glmmTMB).
#
# Built only on existing post-fit accessors (`loglik`, `dof`, `aic`, `nobs`,
# `coef`, `family`) and the public `drm(...)` verb. Adds:
#   * `lrtest` / `anova` — likelihood-ratio test for two nested ML fits.
#   * `aicc`             — small-sample (second-order) AIC correction.
#   * `weights`          — prior observation weights (extends StatsAPI.weights).
#   * `update`           — convenience refit reusing the fitted family.
#
# ML only: REML log-likelihoods are not comparable across different fixed-effect
# structures, so the LR test (and AIC/AICc differences) assume ML fits — which is
# DRModels.jl's default.

using Distributions: Chisq, ccdf
import StatsAPI: weights

"""
    lrtest(reduced::DrmFit, full::DrmFit) -> NamedTuple

Likelihood-ratio test for two **nested**, **ML**-fitted models, mirroring
drmTMB's `anova(reduced, full)`. `reduced` must be a special case of `full`
(fewer parameters); both must be fit by maximum likelihood (DRModels.jl's default —
REML likelihoods are not comparable across fixed-effect structures).

Returns a `NamedTuple` `(; statistic, dof, pvalue)`:

- `statistic = 2 * (loglik(full) - loglik(reduced))` — the LR statistic, which is
  asymptotically `χ²` with `dof` degrees of freedom under the null that the
  reduced model is adequate.
- `dof = dof(full) - dof(reduced)` — the number of extra parameters in `full`.
- `pvalue = ccdf(Chisq(dof), max(statistic, 0))` — the upper-tail χ² p-value.

`dof` must be positive (`full` must have more parameters than `reduced`),
otherwise an `ArgumentError` is thrown. A negative `statistic` (the reduced model
fits *better* — a sign the models are not actually nested, or one did not
converge) is still returned as-is, but the p-value clamps the statistic at zero
(so `pvalue` stays in `[0, 1]`); inspect `statistic` directly in that case.

!!! warning "Variance components are a boundary null"
    The `χ²(dof)` reference is only valid when the extra parameters in `full` are
    interior (regular). When `full` adds a **variance component** that `reduced`
    lacks (a random-effect SD, `:resd`/`:resid`; a Cholesky covariance entry,
    `:recov`; a group-level covariance, `:phylocov`), the tested null (variance =
    0) sits on the **boundary** of the parameter space and the statistic follows a
    chi-bar-square mixture, not `χ²(dof)` (Self & Liang 1987; Stram & Lee 1994).
    The naive `χ²` p-value is then **conservative** (too large — the test loses
    power). `lrtest` detects this case and emits a one-time warning; use
    `lrt_boundary` (or a parametric bootstrap) for a boundary-correct p-value.

# Example
```julia
x = randn(400)
y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(400)
data = (; y, x)

full    = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data)
reduced = drm(bf(@formula(y ~ 1),     @formula(sigma ~ 1)),     Gaussian(); data)

t = lrtest(reduced, full)
t.statistic    # 2·(logLik_full − logLik_reduced), > 0 when x helps
t.dof          # 2 extra parameters (x in μ and in log σ)
t.pvalue       # < 0.05 when x is truly predictive
```
"""
function lrtest(reduced::DrmFit, full::DrmFit)
    _reml_compare_guard(reduced, full, "lrtest")
    _sentinel_compare_guard(reduced, full, "lrtest")
    _marginal_compare_guard(reduced, full, "lrtest")
    _map_compare_guard(reduced, full, "lrtest")
    _nobs_compare_guard(reduced, full, "lrtest")
    Δdof = dof(full) - dof(reduced)
    Δdof > 0 || throw(ArgumentError(
        "lrtest: `full` must have more parameters than `reduced` " *
        "(dof(full) = $(dof(full)), dof(reduced) = $(dof(reduced)); " *
        "did you pass the arguments as (reduced, full)?)"))
    _boundary_vc_warn(reduced, full, "lrtest")
    statistic = 2 * (loglik(full) - loglik(reduced))
    pvalue = ccdf(Chisq(Δdof), max(statistic, 0))
    return (; statistic, dof = Δdof, pvalue)
end

# A fit stranded on the failed-objective sentinel plateau (loglik <= -1e15, see
# `_sentinel_loglik`) has a meaningless likelihood: an LR statistic built from it is
# ~1e18 and its p-value 0 or 1, an AIC ~2e18. Comparison verbs refuse it, like the
# REML/VA refusals; information criteria return NaN with a warning.
function _sentinel_compare_guard(reduced::DrmFit, full::DrmFit, verb::AbstractString)
    for (nm, f) in (("reduced", reduced), ("full", full))
        _sentinel_loglik(f) && throw(ArgumentError(
            "$verb: the `$nm` fit is degenerate (loglik = $(loglik(f)), the failed-" *
            "objective sentinel or non-finite): the optimiser never reached a valid " *
            "likelihood, so a likelihood-ratio comparison is meaningless. Refit " *
            "(different start values / optimiser) before comparing."))
    end
    return nothing
end

function _sentinel_infocrit_nan(fit::DrmFit, which::AbstractString)
    _sentinel_loglik(fit) || return false
    @warn "$which: fit is degenerate (loglik = $(fit.loglik), the failed-objective " *
          "sentinel or non-finite); returning NaN. Refit before using information criteria."
    return true
end

# Block symbols that carry a VARIANCE COMPONENT (random-effect SDs, Cholesky
# covariance entries, group-level covariances). Dropping one of these between
# `reduced` and `full` puts the tested null value (variance = 0) on the BOUNDARY of
# the parameter space, where the LR statistic is NOT χ²(Δdof) — the correct
# reference is a chi-bar-square mixture (see `lrt_boundary`/`chibar_pvalue`). These
# names are shared across univariate AND bivariate fits (e.g. `:phylocov`/`:recov`
# on a bivariate q=2/q=4 phylogenetic fit, same symbol as the univariate case), so
# `_variance_component_blocks`/`_boundary_vc_warn` below already label a bivariate
# fit's boundary variance components correctly (#639) — the per-parameter naming
# gap only affected `_fixed_effect_structure` (:mu1/:mu2/… vs the univariate :mu).
const _VARIANCE_COMPONENT_BLOCKS =
    (:resd, :resid, :recov, :phylocov, :resd_mu, :resd_sigma, :sd, :sd_phylo)

# Variance-component block symbols present in `fit` (with a non-empty range).
function _variance_component_blocks(fit::DrmFit)
    return Symbol[p for (p, r) in fit.blocks
                 if p in _VARIANCE_COMPONENT_BLOCKS && length(r) > 0]
end

# Boundary caveat (issue #304): when `full` carries a variance-component block that
# `reduced` does not, the naive χ²(Δdof) LR reference is INVALID — the tested
# variance sits on the boundary of its space, so the statistic follows a
# chi-bar-square mixture and the χ² p-value is CONSERVATIVE (too large; the test
# loses power). We do NOT silently swap the reference (the mixture order and
# independence assumptions need the caller's knowledge of the design), but we WARN
# and point to the boundary-corrected test. Silent otherwise.
function _boundary_vc_warn(reduced::DrmFit, full::DrmFit, verb::AbstractString)
    dropped = setdiff(_variance_component_blocks(full),
                      _variance_component_blocks(reduced))
    isempty(dropped) && return nothing
    @warn(
        "$verb: `full` adds variance-component block(s) $(dropped) that `reduced` " *
        "lacks, so this compares a variance component against 0 — a BOUNDARY null. " *
        "The reported χ²($(dof(full) - dof(reduced))) p-value is INVALID there " *
        "(conservative: too large, so the test loses power); the correct reference " *
        "is a chi-bar-square mixture. Use `lrt_boundary(full, reduced; q = <#components>)` " *
        "(or a parametric bootstrap) for a boundary-correct p-value.")
    return nothing
end

# Blocks that REML actually RESTRICTS (marginalises/projects out): the response
# MEAN's fixed effects. Univariate `:mu`; bivariate `:mu1`/`:mu2` (see
# gaussian_bivariate.jl's Patterson–Thompson restriction, which "marginalises
# beta_mu1/beta_mu2 only"). A dispersion submodel (`:sigma`/`:sigma1`/`:sigma2`)
# or a correlation submodel (`:rho12`) is estimated INSIDE the restricted
# likelihood as an ordinary (nuisance) parameter, exactly like a variance
# component — comparing REML fits that share the mean design but differ in
# THOSE blocks is the valid, everyday use of REML (e.g. testing a heteroscedastic
# vs homoscedastic error model), not the REML trap.
const _REML_RESTRICTED_MEAN_BLOCKS = (:mu, :mu1, :mu2)

# Fixed-effect (MEAN) structure fingerprint for the REML guard (#639): the
# mean block(s) actually restricted by REML, paired with their coefficient
# names (falling back to the block width when names are absent). On a
# univariate fit this is `:mu`; on a bivariate fit, `:mu1`/`:mu2`. Sorted by
# block symbol so two fits with the same blocks in a different order still
# compare equal.
function _fixed_effect_structure(fit::DrmFit)
    cn = Dict(fit.coefnames)
    fx = Pair{Symbol,Vector{String}}[]
    for (p, r) in fit.blocks
        p in _REML_RESTRICTED_MEAN_BLOCKS || continue
        nms = haskey(cn, p) ? cn[p] : string.(collect(r))
        push!(fx, p => nms)
    end
    sort!(fx; by = first)
    return fx
end

# REML model-selection guard (issue #11, generalized #639): a REML log-likelihood
# is not comparable to an ML (or MAP) log-likelihood AT ALL — they are different
# likelihoods — so any pair with different `estim_method`s is refused outright,
# even when their fixed-effect structures happen to match. Among two REML fits,
# the classic REML trap is comparing DIFFERENT MEAN structures — `:mu`/`:mu1`/
# `:mu2` (generalized to bivariate fits by `_fixed_effect_structure`) — the
# restricted likelihoods are built on different error-contrast bases and are not
# comparable. Comparing REML fits that differ only in a dispersion/correlation
# submodel (`:sigma`/`:sigma1`/`:sigma2`/`:rho12`) or a variance-component
# structure, with the SAME mean design, is valid — that submodel is a nuisance
# parameter inside the restricted likelihood, not something REML restricts away.
# ML-vs-ML is always fine. We ERROR on both invalid cases (the LR test would be
# meaningless) and stay silent otherwise.
function _reml_compare_guard(a::DrmFit, b::DrmFit, verb::AbstractString)
    (a.estim_method === :REML || b.estim_method === :REML) || return nothing
    if a.estim_method !== b.estim_method
        throw(ArgumentError(
            "$verb: cannot compare fits estimated by different methods " *
            "(estim_method = :$(a.estim_method) vs :$(b.estim_method)) — a REML " *
            "log-likelihood is not comparable to an ML (or MAP) log-likelihood, " *
            "even when their fixed-effect structures match: they are different " *
            "likelihoods. Refit both with the same `method` (`:ML` to compare across " *
            "fixed-effect structures, or `:REML` — with identical fixed-effect " *
            "structure in every mean/scale block — to compare variance components)."))
    end
    if _fixed_effect_structure(a) != _fixed_effect_structure(b)
        throw(ArgumentError(
            "$verb: cannot compare REML fits with different fixed-effect structure in " *
            "any mean or scale block — REML log-likelihoods are not comparable across " *
            "fixed-effect structures (only across variance-component structure). Refit " *
            "both with method = :ML for a cross-structure test."))
    end
    return nothing
end

# Penalized-MAP guard (A4c): a penalized fit is a maximum-a-posteriori estimate,
# so its variance components are deliberately shrunk toward zero. The likelihood
# ratio of two such fits does not have the usual chi-square reference
# distribution — the prior is doing part of the work the test would attribute to
# the data. drmTMB flags the same hazard as a note from `check_penalized_fit()`;
# because a silent wrong p-value is worse than a refusal, DRModels.jl errors here and
# surfaces the same information through `check_drm(fit).penalized_map`.
# `update` of a missing-response or imputed fit can refit a different set of
# rows than the seed. The two log-likelihoods are then not a nested comparison
# (#1002). Refuse that pair rather than returning a statistic on mixed samples.
function _nobs_compare_guard(a::DrmFit, b::DrmFit, verb::AbstractString)
    nobs(a) == nobs(b) && return nothing
    throw(ArgumentError(
        "$verb: the two fits were estimated on different samples " *
        "(nobs = $(nobs(a)) vs $(nobs(b))). A likelihood-ratio comparison " *
        "requires the same observations, which a missing-response or imputed " *
        "`update` does not guarantee. Refit both on the same rows."))
end

function _map_compare_guard(a::DrmFit, b::DrmFit, verb::AbstractString)
    (a.estim_method === :MAP || b.estim_method === :MAP) || return nothing
    throw(ArgumentError(
        "$verb: cannot compare penalized (MAP) fits — a `penalty = drm_phylo_penalty(...)` " *
        "fit shrinks its variance components, so the likelihood-ratio statistic does not have " *
        "the usual chi-square distribution. Refit both without `penalty` to test, or compare " *
        "the penalized fits on their own terms."))
end

# A fit whose formula has no random effect integrates nothing, so its `loglik` is
# the exact log-likelihood whatever `marginal` tag it carries (a fixed-effects fit
# is tagged `:LA` by default). Conservative: a bivariate formula, attached BLUPs, a
# variance-component block, an `sd(g) ~ …` formula, or any random-effect /
# structured / `meta_V` marker counts as "has a random effect".
function _fit_is_re_free(fit::DrmFit)
    f = fit.formula
    (f isa DrmFormula && fit.ranef === nothing &&
        isempty(_variance_component_blocks(fit))) || return false
    for (name, rhs) in f.forms
        startswith(String(name), "sd") && return false           # sd(g) ~ … / sd(g, phylogenetic) ~ …
        for t in (rhs isa Tuple ? rhs : (rhs,))
            t isa FunctionTerm && any(m -> t.f === m, (|, meta_V, relmat, animal, phylo, spatial)) &&
                return false
        end
    end
    return true
end

# Mixed-marginal guard (#136 Arc 0): fits whose random-effect integrals were
# approximated differently are not comparable in lrtest / anova. A VA `loglik` is
# an ELBO, not a log-likelihood; `:LA` (GHQ-32 on an ordinary `(1 | g)`) and
# `:Laplace` (one-point Laplace) differ by integration error. The one safe mixed
# pair is a random-effect-free fit (exact log-likelihood) against a non-VA fit,
# e.g. the fixed-effects model against a `marginal = :Laplace` random intercept.
function _marginal_compare_guard(a::DrmFit, b::DrmFit, verb::AbstractString)
    a.marginal === b.marginal && return nothing
    no_va = a.marginal !== :VA && b.marginal !== :VA
    no_va && (_fit_is_re_free(a) || _fit_is_re_free(b)) && return nothing
    reason = if !no_va
        "the VA objective is an ELBO, not a log-likelihood (#136)"
    else
        detail = Set((a.marginal, b.marginal)) == Set((:LA, :Laplace)) ?
            " (`:LA` is GHQ-32 on an ordinary `(1 | g)`; `:Laplace` is the one-point Laplace approximation)" : ""
        "the two approximate the random-effect integral differently$detail, so their " *
        "log-likelihood difference would mix model fit with integration error"
    end
    throw(ArgumentError(
        "$verb: cannot compare fits with different marginal approximations " *
        "(`:$(a.marginal)` vs `:$(b.marginal)`): $reason. Refit both with the same `marginal`."))
end

"""
    anova(reduced::DrmFit, full::DrmFit) -> NamedTuple

Alias for [`lrtest`](@ref), matching drmTMB's `anova(reduced, full)` spelling for
a nested likelihood-ratio test. Returns the same
`(; statistic, dof, pvalue)` NamedTuple.

# Example
```julia
anova(reduced, full) == lrtest(reduced, full)   # true
```
"""
anova(reduced::DrmFit, full::DrmFit) = lrtest(reduced, full)

"""
    aicc(fit::DrmFit) -> Float64

Corrected Akaike information criterion (AICc) — the small-sample, second-order
correction to [`aic`](@ref):

    AICc = AIC + 2k(k + 1) / (n − k − 1)

with `k = dof(fit)` estimated parameters and `n = nobs(fit)` observations. The
correction is always positive, so `aicc(fit) ≥ aic(fit)`, and it converges to
`aic(fit)` as `n → ∞`. Prefer AICc over AIC when `n / k` is small (a common rule
of thumb is `n / k < 40`). Like AIC, AICc compares models fit by **ML** on the
same data; lower is better.

If `n - k - 1 <= 0` (too few observations for the correction to be defined),
returns `Inf`. On a **VA** fit this errors before that short-circuit: `loglik`
carries an ELBO, not a marginal log-likelihood.

# Example
```julia
fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data)
aicc(fit) > aic(fit)        # the correction is strictly positive
isfinite(aicc(fit))         # finite whenever n - k - 1 > 0
```
"""
function aicc(fit::DrmFit)
    _va_infocrit_guard(fit, "aicc")
    _sentinel_infocrit_nan(fit, "aicc") && return NaN
    k = dof(fit)
    n = nobs(fit)
    n - k - 1 > 0 || return Inf
    return aic(fit) + 2 * k * (k + 1) / (n - k - 1)
end

"""
    weights(fit::DrmFit) -> Vector{Float64}

Prior (per-observation) weights used in the fit — drmTMB / glmmTMB's `weights()`.
DRModels.jl fits do not currently store prior weights, so this returns
`ones(nobs(fit))` (every observation weighted equally). Extends
`StatsAPI.weights`.

# Example
```julia
fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data)
weights(fit) == ones(nobs(fit))   # all-ones prior weights
```
"""
weights(fit::DrmFit) = ones(nobs(fit))

"""
    update(fit::DrmFit, formula, args...; data, kwargs...) -> DrmFit

Refit `fit` with a new `formula` (a [`bf`](@ref) bundle), reusing the fitted
family and every estimation option the fit stores:

- `method` — `:REML` when `fit.estim_method === :REML` (the default `:ML` is
  `drm`'s own default and is not repeated);
- `marginal` — when it is not the default `:LA`;
- `penalty` — the stored [`drm_phylo_penalty`](@ref) specification of a `:MAP` fit.

Explicit keywords override those stored options (`update(fit, formula; data,
method = :ML)` refits a REML seed by maximum likelihood). Changing `method`
on a penalized MAP fit drops the stored penalty unless `penalty` is passed
again; otherwise `method = :ML` would still be a MAP refit. `family` is not a
refit keyword: the fitted family is kept, and passing one is an error. An
unnamed extra positional argument is an error: a keyword-only refit would
otherwise drop it and silently ignore the value.

`data` must be supplied. A `DrmFit` does not retain its observations, nor the
options that were only arguments to the original `drm` call: `K`, `A`, `tree`,
`coords`, `algorithm`, `g_tol`, `profile_ci`, `phylo_coupled`, `sparse`,
`impute`, and `missing`. Pass those again as keywords when the refit needs them.

# Example
```julia
full    = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data)
# Drop x everywhere, keeping the same Gaussian family and the seed estimator:
reduced = update(full, bf(@formula(y ~ 1), @formula(sigma ~ 1)); data = data)
length(coef(reduced)) < length(coef(full))   # fewer parameters
```
"""
function update(fit::DrmFit, formula, args...; data, kwargs...)
    isempty(args) || throw(ArgumentError(
        "update requires named arguments. Pass values as `data = ...`, " *
        "`method = ...`, or another named argument; an unnamed extra is dropped " *
        "by a keyword-only refit and would silently refit without it " *
        "(got $(length(args)) unnamed)."))
    if haskey(kwargs, :family)
        throw(ArgumentError(
            "update keeps the fitted family. `family` is not a refit keyword; " *
            "call `drm(formula, family; data = ...)` to fit a different family " *
            "(got family = $(repr(kwargs[:family])))."))
    end
    stored = _update_stored_options(fit)
    # A caller who changes the estimator does not want the old MAP penalty.
    # Passing `penalty` explicitly still overrides that drop.
    caller_method = get(kwargs, :method, nothing)
    if caller_method !== nothing && caller_method != fit.estim_method &&
            haskey(stored, :penalty) && !haskey(kwargs, :penalty)
        stored = Base.structdiff(stored, NamedTuple{(:penalty,)})
    end
    merged = merge(stored, NamedTuple(kwargs))
    return drm(formula, fit.family; data = data, merged...)
end

# Estimation options `DrmFit` actually stores. `:ML` and `:LA` are `drm`
# defaults, so repeating them would be a no-op; a `:MAP` fit is repeated by
# its stored penalty, not by `method = :MAP` (the fitter rejects that symbol).
function _update_stored_options(fit::DrmFit)
    kw = Pair{Symbol,Any}[]
    if fit.estim_method === :REML
        push!(kw, :method => :REML)
    elseif fit.estim_method === :MAP
        fit.penalty === nothing && throw(ArgumentError(
            "update: this fit is marked :MAP but does not store a penalty " *
            "specification, so the refit cannot repeat the penalized estimator."))
    elseif fit.estim_method !== :ML
        throw(ArgumentError(
            "update: unsupported estim_method :$(fit.estim_method)"))
    end
    fit.penalty === nothing || push!(kw, :penalty => fit.penalty)
    fit.marginal === :LA || push!(kw, :marginal => fit.marginal)
    return NamedTuple(kw)
end
