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
# `lrtest` is StatsModels' function: DRModels adds methods to it (on `DrmFit` only) and
# re-exports the same name, so `using DRModels, StatsModels; lrtest(...)` is not an
# ambiguous binding on Julia 1.12 and later.
import StatsModels: lrtest

"""
    lrtest(reduced::DrmFit, full::DrmFit; mixture::Bool = false) -> NamedTuple

Likelihood-ratio test for two **nested**, **ML**-fitted models, mirroring
drmTMB's `anova(reduced, full)`. `reduced` must be a special case of `full`
(fewer parameters); both must be fit by maximum likelihood (DRModels.jl's default —
REML likelihoods are not comparable across fixed-effect structures).

This is a method of `StatsModels.lrtest`, the same function (`DRModels.lrtest ===
StatsModels.lrtest`), so `using DRModels, StatsModels` and an unqualified `lrtest(...)`
work together. The method applies to `DrmFit` pairs only.

Returns a `NamedTuple` `(; statistic, dof, pvalue)`:

- `statistic = 2 * (loglik(full) - loglik(reduced))` — the LR statistic, which is
  asymptotically `χ²` with `dof` degrees of freedom under the null that the
  reduced model is adequate.
- `dof = dof(full) - dof(reduced)` — the number of extra parameters in `full`.
- `pvalue = ccdf(Chisq(dof), max(statistic, 0))` — the upper-tail χ² p-value.

`dof` must be positive (`full` must have more parameters than `reduced`),
otherwise an `ArgumentError` is thrown. Both fits must have been estimated on
the same observation rows, not only the same `nobs`: two fits that drop
different rows and happen to keep the same count are refused (#1053). A
negative `statistic` (the reduced model fits *better*, a sign the models are
not actually nested, or one did not converge) is still returned as-is, but the
p-value clamps the statistic at zero (so `pvalue` stays in `[0, 1]`); inspect
`statistic` directly in that case.

!!! warning "Variance components and Poisson ⊂ NB2 are a boundary null"
    The `χ²(dof)` reference is only valid when the extra parameters in `full` are
    interior (regular). When `full` adds a **variance component** that `reduced`
    lacks (a random-effect SD, `:resd`/`:resid`; a Cholesky covariance entry,
    `:recov`; a group-level covariance, `:phylocov`), or when `full` is
    `NegBinomial2()` and `reduced` is `Poisson()` (the extra `sigma` tested at
    0), the tested null sits on the **boundary** of the parameter space and the
    statistic follows a chi-bar-square mixture, not `χ²(dof)` (Self & Liang
    1987; Stram & Lee 1994). The naive `χ²` p-value is then **conservative**
    (too large — the test loses power). `lrtest` detects these cases and emits
    a one-time warning; use `lrt_boundary` (or a parametric bootstrap) for a
    boundary-correct p-value. This function does **not** swap the reported
    p-value (#304). drmTMB's `anova()` does swap in the mixture for the same
    pairs; call `lrt_boundary` here to match that p-value.

## Opt-in mixture reference: `mixture = true`
`lrtest(reduced, full; mixture = true)` reports the boundary-correct p-value when
`full` differs from `reduced` by **exactly one** variance component (for example
`(1 | g)` left out of `reduced`, or `(1 | g) + (0 + x | g)` against `(1 | g)`) or by
the one `sigma` of `Poisson() ⊂ NegBinomial2()`. The reference is then the equal
mixture of `χ²(0)` and `χ²(1)`, so `pvalue = 0.5 * P(χ²₁ > statistic)` (Self &
Liang 1987; Stram & Lee 1994). The result gains two fields: `pvalue_naive` (the
plain `χ²(dof)` p-value) and `reference = :chibar_0_1`; the boundary warning is
not emitted, because the caveat is handled. The default stays `mixture = false`
(plain `χ²`, with the warning), so existing calls do not change.

The equal mixture of `χ²(0)` and `χ²(1)` is the correct null for this test. The tested
variance sits at the edge of its range, so the statistic is exactly zero in about half of
the data sets, and only the other half follows `χ²(1)`. The plain `χ²(1)` is therefore
conservative: whenever the statistic is positive its p-value is about twice the mixture
p-value, and the plain test rejects less often than its nominal level. The mixture stays
opt-in here, and the plain test is still the default. A test of a correlation alone is
refused under `mixture = true` (see below), because the mixture does not apply to it.

`mixture = true` is refused with an `ArgumentError`, by name, unless `full` adds
exactly one variance (SD) parameter, no correlation parameter, and the fixed
effects of the two fits are identical (or the pair is Poisson ⊂ NegBinomial2 with
the same fixed effects). It is refused for a difference in fixed effects or in a
family other than Poisson ⊂ NegBinomial2 (the plain `χ²` is the right test there),
for more than one added variance parameter, and for a pair that adds a correlation:
a random slope together with its correlation, for example `(1 + x | g)` against
`(1 | g)`, adds a variance and a correlation (its reference is a different
mixture), and `(1 | g) + (0 + x | g)` against `(1 + x | g)` adds only the
correlation, whose null value is interior, so the plain `χ²(1)` is the right
test. Use a parametric bootstrap for the first of these.
A slope on the log standard deviation is an interior null, not a variance at
zero, so `sd(g) ~ z` against `sd(g) ~ 1`, and `(1 | g)` against `sd(g) ~ z`,
are refused. The one added standard deviation is matched by its covariance
structure and its grouping, not by the raw coefficient string and not by count
alone: a one-group `sd:(Intercept)`, a plain `resd:g`, and a two-group
`sd:g: (Intercept)` are the same iid component on `g`, and `sd_phylo` is the
same component as a phylogenetic intercept on that grouping. An iid intercept
and a phylogenetic, spatial, animal, or relmat intercept on the same grouping
are different components, so `phylo(1 | sp)` against `(1 | sp) + (1 | g)` is
refused. `(1 | g)` against `(1 | h) + (1 | k)` is refused when the statistic
is positive. A clearly negative statistic on that pair is still reported as
not nested, before the label check.
`mixture = true` also refuses a clearly negative statistic, one below
`-1e-6 * max(1, |loglik(full)|, |loglik(reduced)|)`, as "not nested or a fit did
not converge", instead of returning `pvalue = 0.5`. A statistic within that margin
of zero is a boundary result and is accepted. The plain test (`mixture = false`)
still returns a negative statistic as it is, with the statistic clamped at zero in the p-value.
With few groups the mixture is conservative: in null data sets (Gaussian
`(1 | g)`, no group variance, 10 rows per group) the mixture p-value rejected at
the 0.05 level in 2.9% of 3000 data sets with 20 groups and in 4.1% of 5000 with
100 groups, against 1.6% for the plain `χ²(1)` in both. About 55% to 60% of the
statistics sit on the `χ²(0)` atom, the share of data sets where the estimated
group variance is zero. lme4 (`lmer(REML = FALSE)` against `lm`) behaves the same
on the same kind of data (2.2% and 4.5% in 1000 data sets each), and the
statistics agree with lme4 to 1e-8.
`lrt_boundary` is the separate function that applies the 50:50 or the two-
independent-components mixture to a pair you name yourself.

References: Self, S. G. and Liang, K.-Y. (1987), Asymptotic properties of maximum
likelihood estimators and likelihood ratio tests under nonstandard conditions,
*Journal of the American Statistical Association* 82, 605-610. Stram, D. O. and Lee,
J. W. (1994), Variance components testing in the longitudinal mixed effects model,
*Biometrics* 50, 1171-1177.

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
function lrtest(reduced::DrmFit, full::DrmFit; mixture::Bool = false)
    _reml_compare_guard(reduced, full, "lrtest")
    _sentinel_compare_guard(reduced, full, "lrtest")
    _marginal_compare_guard(reduced, full, "lrtest")
    _map_compare_guard(reduced, full, "lrtest")
    _nobs_compare_guard(reduced, full, "lrtest")
    _unconverged_compare_warn(reduced, full, "lrtest")
    Δdof = dof(full) - dof(reduced)
    Δdof > 0 || throw(ArgumentError(
        "lrtest: `full` must have more parameters than `reduced` " *
        "(dof(full) = $(dof(full)), dof(reduced) = $(dof(reduced)); " *
        "did you pass the arguments as (reduced, full)?)"))
    statistic = 2 * (loglik(full) - loglik(reduced))
    if mixture
        # `:variance` is the one-SD count gate; `:family` is Poisson ⊂ NegBinomial2.
        # The negative-statistic refusal stays ahead of the label check so a
        # swapped grouping whose statistic is negative keeps that message.
        gate = _single_variance_boundary_guard(reduced, full, Δdof)
        # A clearly negative statistic means `reduced` fits better: the pair is not
        # nested or a fit did not converge. The mixture would report p = 0.5 for it.
        scale = max(1.0, abs(loglik(full)), abs(loglik(reduced)))
        statistic < -1e-6 * scale && throw(ArgumentError(
            "lrtest: `mixture = true` got a negative statistic ($statistic): `reduced` " *
            "fits clearly better than `full`, so the pair is not nested or a fit did " *
            "not converge. Check both fits (`reduced` must be `full` with one " *
            "variance component removed) and refit before testing."))
        gate === :variance && _matching_variance_labels_guard(reduced, full)
        pvalue_naive = ccdf(Chisq(Δdof), max(statistic, 0))
        return (; statistic, dof = Δdof, pvalue = chibar_pvalue(statistic, 1),
                  pvalue_naive, reference = :chibar_0_1)
    end
    _boundary_vc_warn(reduced, full, "lrtest")
    pvalue = ccdf(Chisq(Δdof), max(statistic, 0))
    return (; statistic, dof = Δdof, pvalue)
end

# `lrtest(...; mixture = true)` applies the 0.5 χ²(0) + 0.5 χ²(1) reference, which is
# right only when `full` adds exactly ONE variance (SD) parameter, no correlation
# parameter, and nothing else. Accept: every non-variance block has the same
# coefficient names in both fits, `full` has one more SD parameter and the same
# number of correlation parameters, and Δdof is 1; or the Poisson ⊂ NegBinomial2
# nest with the same fixed effects. Returns `:variance` or `:family` so the caller
# can match SD labels only on the variance gate. Refuse everything else by name.
# A correlation is not a boundary parameter (its null 0 is interior), so a pair
# such as `(1 | g) + (0 + x | g)` against `(1 + x | g)` is refused: the plain
# χ²(1) is right. A slope on an `sd` / `sd_phylo` block is an interior null too:
# only the `(Intercept)` column of those blocks is a variance.
function _single_variance_boundary_guard(reduced::DrmFit, full::DrmFit, Δdof::Integer)
    nonvc(f) = Dict(p => (length(r), String.(something(_block_coefnames(f, p), String[])))
                    for (p, r) in f.blocks
                    if !(p in _VARIANCE_COMPONENT_BLOCKS) && length(r) > 0)
    same_fixed = nonvc(full) == nonvc(reduced)
    if _family_boundary_nest(reduced, full) !== nothing && Δdof == 1
        fr, ff = nonvc(reduced), nonvc(full)
        all(haskey(ff, p) && ff[p] == v for (p, v) in fr) && return :family
    end
    sd_params, corr_params = _sd_and_correlation_counts(full)
    sd_red, corr_red = _sd_and_correlation_counts(reduced)
    Δsd, Δcorr = sd_params - sd_red, corr_params - corr_red
    if Δsd == 1 && Δcorr == 0 && Δdof == 1 && same_fixed
        return :variance
    end
    reason = if Δsd <= 0 && Δcorr <= 0 && same_fixed &&
               !_same_string_multiset(_logsd_slope_names(reduced), _logsd_slope_names(full))
        "`full` changes a slope of a log standard deviation (for example `sd(g) ~ 1` " *
        "against `sd(g) ~ z`, or `(1 | g)` against `sd(g) ~ z`); that slope is an " *
        "interior null, so the plain χ² reference is the right test"
    elseif Δsd <= 0 && Δcorr <= 0 && same_fixed
        "`full` adds no variance component to `reduced` (the difference is in the " *
        "family or in a block that is not a variance), so the plain χ² reference " *
        "is the right test"
    elseif !same_fixed
        "`full` and `reduced` differ in their fixed effects or in a non-variance " *
        "block, so the plain χ² reference is the right test"
    elseif Δcorr != 0 && Δsd <= 0
        "`full` adds only $(Δcorr) correlation parameter(s) (for example `(1 | g) + " *
        "(0 + x | g)` against `(1 + x | g)`); a correlation is tested at an " *
        "interior null, so the plain χ² reference is the right test"
    elseif Δcorr != 0
        "`full` adds $Δsd variance parameter(s) and $Δcorr correlation parameter(s) " *
        "(for example a random slope with its correlation); the reference for that " *
        "is a different mixture, which is not implemented"
    elseif Δdof > 1 || Δsd > 1
        "`full` adds $Δsd variance parameters and $Δdof parameters in all; the " *
        "reference for that is a different mixture, which is not implemented"
    else
        "`full` and `reduced` differ in more than the one variance component"
    end
    throw(ArgumentError(
        "lrtest: `mixture = true` is only for a pair that differs by exactly one " *
        "variance component (reference 0.5 χ²(0) + 0.5 χ²(1)); $reason. " *
        "Call `lrtest(reduced, full)` for the plain test, or use a parametric bootstrap."))
end

# Number of SD parameters and of correlation parameters in the variance-component
# blocks of `fit`. `:recov` and `:phylocov` hold the Cholesky entries of a q x q
# covariance, q (q + 1) / 2 of them: q SDs and q (q - 1) / 2 correlations. An
# `:sd` / `:sd_phylo` block is a regression on the log standard deviation: only
# its `(Intercept)` columns are variances (a slope is an interior null). Every
# other variance-component block counts one SD per coefficient. A width that is
# not triangular cannot be split, so it counts as all SD (the old behaviour).
function _sd_and_correlation_counts(fit::DrmFit)
    nsd = 0
    ncorr = 0
    for (p, r) in fit.blocks
        p in _VARIANCE_COMPONENT_BLOCKS || continue
        w = length(r)
        if p === :recov || p === :phylocov
            q = (isqrt(8w + 1) - 1) ÷ 2
            if q * (q + 1) ÷ 2 == w
                nsd += q
                ncorr += w - q
                continue
            end
        end
        if p === :sd || p === :sd_phylo
            names = _block_coefnames(fit, p)
            if names !== nothing && length(names) == w
                nsd += count(n -> _is_sd_intercept_name(String(n)), names)
                continue
            end
        end
        nsd += w
    end
    return nsd, ncorr
end

# `(Intercept)`, including a disambiguated `"g: (Intercept)"` column.
_is_sd_intercept_name(name::AbstractString) =
    name == "(Intercept)" || endswith(name, "(Intercept)")

# Non-intercept columns of `:sd` / `:sd_phylo`: slopes of the log standard deviation.
function _logsd_slope_names(fit::DrmFit)
    out = String[]
    for (p, r) in fit.blocks
        (p === :sd || p === :sd_phylo) || continue
        names = _block_coefnames(fit, p)
        (names === nothing || length(names) != length(r)) && continue
        for n in names
            _is_sd_intercept_name(String(n)) || push!(out, String(n))
        end
    end
    return out
end

function _string_counts(xs)
    d = Dict{String,Int}()
    for s in xs
        d[s] = get(d, s, 0) + 1
    end
    return d
end
_same_string_multiset(a, b) = _string_counts(a) == _string_counts(b)

# Labels present in `full` and not in `reduced`, or `nothing` when `reduced`
# is not a submultiset of `full` (a component was swapped, not added).
function _added_string_labels(reduced, full)
    d = _string_counts(full)
    for s in reduced
        c = get(d, s, 0)
        c == 0 && return nothing
        d[s] = c - 1
    end
    extra = String[]
    for (s, c) in d
        for _ in 1:c
            push!(extra, s)
        end
    end
    return extra
end

# Diagonal Cholesky entry (`:L11`) is an SD; an off-diagonal (`:L21`) is a correlation.
function _cholesky_role(name::AbstractString)
    m = match(r":L(\d+)(\d+)$", name)
    m !== nothing && return m.captures[1] == m.captures[2] ? :sd : :cor
    n = lowercase(name)
    (occursin("atanh", n) || occursin("rho", n) || occursin(":cor", n)) && return :cor
    (occursin("log_sd", n) || occursin(":sd", n)) && return :sd
    return :unknown
end

function _push_vc_labels!(sd::Vector{String}, cor::Vector{String}, p::Symbol, names, w::Int)
    label(i) = length(names) == w ? string(p, ":", names[i]) : string(p, "#", i)
    if p === :recov || p === :phylocov
        q = (isqrt(8w + 1) - 1) ÷ 2
        if q * (q + 1) ÷ 2 == w && length(names) == w
            roles = [_cholesky_role(names[i]) for i in 1:w]
            if all(r -> r !== :unknown, roles) && count(==(:sd), roles) == q
                for i in 1:w
                    push!(roles[i] === :sd ? sd : cor, label(i))
                end
                return nothing
            end
            for i in 1:q
                push!(sd, label(i))
            end
            for i in (q + 1):w
                push!(cor, label(i))
            end
            return nothing
        end
    end
    if (p === :sd || p === :sd_phylo) && length(names) == w
        for i in 1:w
            _is_sd_intercept_name(names[i]) && push!(sd, label(i))
        end
        return nothing
    end
    for i in 1:w
        push!(sd, label(i))
    end
    return nothing
end

# SD labels and correlation labels, one string per parameter the count above
# treats as a variance or a correlation. Used to refuse a pair whose counts
# differ by one SD but whose labels were swapped (`(1 | g)` against `(1 | h) + (1 | k)`).
function _vc_sd_and_cor_labels(fit::DrmFit)
    sd = String[]
    cor = String[]
    for (p, r) in fit.blocks
        p in _VARIANCE_COMPONENT_BLOCKS || continue
        w = length(r)
        w == 0 && continue
        raw = _block_coefnames(fit, p)
        names = raw === nothing ? String[] : String.(raw)
        _push_vc_labels!(sd, cor, p, names, w)
    end
    return sd, cor
end

# The same random-effect SD is not stored under one spelling. A plain `(1 | g)`
# is `resd:g`. One `sd(g) ~ 1` is `sd:(Intercept)`, with the group only on the
# formula. Two groups in that block are `sd:g: (Intercept)` and `sd:h: (Intercept)`,
# and a plain `(1 | h)` moves into the same block once any `sd` formula is present.
# `sd_phylo:(Intercept)` and a phylogenetic `resd` intercept are the same kind of
# split. The canonical key is `structure:grouping` (`iid:g`, `phylo:species`),
# so a structured intercept does not alias an iid intercept on the same grouping.
# The raw coefficient strings stay in the refusal message.
#
# Structure comes from the fitted term, not from the spelling of the coefficient.
# An `:sd_phylo` block is phylogenetic. An `:sd` block reads `sd_targets` (so
# `sd(g, spatial)` is `spatial`, and a plain bar pulled into that block, which
# has no target, is `iid`). A `:resd` name is matched to the formula's ordinary
# bars and structured markers. Where an ordinary intercept shares the marker's
# grouping, the structured-plus-ranef route stores the ordinary one as `g_iid`
# and the marker as `g` (`_drm_gaussian_structured_plus_ranef`); that label is
# emitted only for a collision the formula actually contains. A temporal fit's
# `ranef.temporal` record is used instead: its bare group is the temporal SD,
# and the stable field is `g_iid` or `g_phylo`.
_sd_structure_token(level::Symbol) = if level === :phylogenetic
    "phylo"
elseif level === :phylogenetic_interaction
    "phylo_interaction"
else
    String(level)
end

_sd_key(structure::AbstractString, group::AbstractString) = string(structure, ":", group)

function _canonical_sd_label(fit::DrmFit, label::AbstractString)
    i = findfirst(':', label)
    i === nothing && return String(label)
    block = label[1:prevind(label, i)]
    rest = label[nextind(label, i):end]
    if block == "sd" || block == "sd_phylo"
        group = _sd_intercept_group(fit, block, rest)
        group === nothing && return String(label)
        structure = _sd_block_structure(fit, block, group)
        structure === nothing && return String(label)
        return _sd_key(structure, group)
    elseif block == "resd"
        return _canonical_resd_name(fit, rest)
    end
    return String(label)
end

# `"g: (Intercept)"` carries its group. A bare `"(Intercept)"` is the only
# component of that block, so the group comes from the formula. `nothing` when
# this coefficient is not an intercept the formula can name.
function _sd_intercept_group(fit::DrmFit, block::AbstractString, rest::AbstractString)
    suffix = ": (Intercept)"
    if endswith(rest, suffix)
        return chop(rest; tail = length(suffix))
    end
    rest == "(Intercept)" || return nothing
    return _bare_sd_intercept_group(fit, block)
end

# `:sd_phylo` is phylogenetic. A plain `:sd` column is the level of the one
# `sd()` target on that grouping, or `iid` when the column is an ordinary bar
# with no target of its own. `nothing` when several targets disagree.
function _sd_block_structure(fit::DrmFit, block::AbstractString, group::AbstractString)
    block == "sd_phylo" && return "phylo"
    f = fit.formula
    f isa DrmFormula || return "iid"
    levels = Symbol[]
    for item in _sd_target_parts(f)
        t = item.target
        String(t.group) == group || continue
        push!(levels, t.level)
    end
    unique!(levels)
    isempty(levels) && return "iid"
    return length(levels) == 1 ? _sd_structure_token(only(levels)) : nothing
end

function _canonical_resd_name(fit::DrmFit, name::AbstractString)
    temporal = _temporal_sd_keys(fit)
    if temporal !== nothing
        return get(temporal, String(name), String(name))
    end
    keyed = _formula_resd_keys(fit)
    keyed === nothing && return String(name)
    return get(keyed, String(name), String(name))
end

# `ranef.temporal` is the fitted temporal term. The bare group name is that
# term's SD. A stable ordinary intercept is `g_iid`; a paired phylogenetic
# field is `g_phylo`. The two stable fields are never both present.
function _temporal_sd_keys(fit::DrmFit)
    re = fit.ranef
    (re isa NamedTuple && hasproperty(re, :temporal)) || return nothing
    info = re.temporal
    hasproperty(info, :group) || return nothing
    g = String(info.group)
    out = Dict{String,String}(g => _sd_key("temporal", g))
    if get(info, :has_ordinary, false) === true
        out[g * "_iid"] = _sd_key("iid", g)
    elseif get(info, :phylo, nothing) !== nothing
        out[g * "_phylo"] = _sd_key("phylo", g)
    end
    return out
end

# Coefficient name → canonical key, from the mean formula's ordinary bars and
# structured markers (`phylo` / `spatial` / `animal` / `relmat`). A name this
# does not predict is left out, and the caller keeps the raw string.
function _formula_resd_keys(fit::DrmFit)
    f = fit.formula
    f isa DrmFormula || return nothing
    mu = _formula_mu_rhs(f)
    mu === nothing && return nothing
    local split
    try
        split = _split_ranef(mu;
            allow_phylo_slope = true, allow_phylo_slope_only = true,
            allow_animal_slope = true, allow_animal_intercept_slope = true,
            allow_relmat_slope = true, allow_relmat_intercept_slope = true,
            allow_spatial_slope = true, allow_spatial_intercept_slope = true,
            allow_temporal = true)
    catch
        return nothing
    end
    _, re, _, structured, structured_slope, _, animal_slope = split
    markers = _collect_structured(mu)
    structured_groups = Set(last(marker) for marker in markers)
    out = Dict{String,String}()
    for (lhs, grp) in re
        kind_var = try
            _re_kind(lhs)
        catch
            continue
        end
        kind, var = kind_var
        if kind === :intercept
            # Collision with a structured marker on this grouping: the ordinary
            # intercept is stored as `g_iid`, the marker keeps the bare group.
            label = grp in structured_groups ? string(grp, "_iid") : String(grp)
            out[label] = _sd_key("iid", String(grp))
        elseif kind === :slope && var !== nothing
            label = string(grp, ":", var)
            out[label] = _sd_key("iid", label)
        end
    end
    for (kind, grp) in markers
        label = String(grp)
        haskey(out, label) || (out[label] = _sd_key(_sd_structure_token(kind), label))
    end
    if structured !== nothing
        slope = structured_slope !== nothing ? structured_slope : animal_slope
        kind, grp = structured
        if slope !== nothing
            label = string(grp, ":", slope)
            haskey(out, label) || (out[label] = _sd_key(_sd_structure_token(kind), label))
        end
    end
    return out
end

function _formula_mu_rhs(f::DrmFormula)
    for (p, rhs) in f.forms
        p === :mu && return rhs
    end
    return nothing
end

# Group of a one-component `:sd` or `:sd_phylo` intercept. `nothing` when the
# formula does not name exactly one group for that block; the raw label is then
# left as it is, so two unnamed intercepts cannot alias each other.
function _bare_sd_intercept_group(fit::DrmFit, block::AbstractString)
    f = fit.formula
    f isa DrmFormula || return nothing
    if block == "sd_phylo"
        parts = _sdphylo_parts(f)
        length(parts) == 1 && return String(first(parts)[1])
    end
    mu = _formula_mu_rhs(f)
    mu === nothing && return nothing
    local re, structured
    try
        _, re, _, structured = _split_ranef(mu)
    catch
        return nothing
    end
    if block == "sd_phylo"
        structured !== nothing && structured[1] === :phylo && return String(structured[2])
        return nothing
    end
    block == "sd" || return nothing
    groups = unique(String(grp) for (_, grp) in re)
    length(groups) == 1 && return only(groups)
    # `sd(g, spatial)` (and animal / relmat) has no ordinary `(1 | g)` bar.
    # The one `sd()` target names the grouping.
    targets = _sd_target_parts(f)
    return length(targets) == 1 ? String(only(targets).target.group) : nothing
end

_canonical_sd_labels(fit::DrmFit, labels) = [_canonical_sd_label(fit, s) for s in labels]

function _matching_variance_labels_guard(reduced::DrmFit, full::DrmFit)
    sd_r, cor_r = _vc_sd_and_cor_labels(reduced)
    sd_f, cor_f = _vc_sd_and_cor_labels(full)
    extra = _added_string_labels(_canonical_sd_labels(reduced, sd_r),
                                 _canonical_sd_labels(full, sd_f))
    if extra !== nothing && length(extra) == 1 && _same_string_multiset(cor_r, cor_f)
        return nothing
    end
    throw(ArgumentError(
        "lrtest: `mixture = true` is only for a pair that differs by exactly one " *
        "variance component, matched by label (the same components, plus one standard " *
        "deviation). `reduced` has variance labels [$(join(sort(sd_r), ", "))] and " *
        "`full` has [$(join(sort(sd_f), ", "))], so this is not that pair " *
        "(for example `(1 | g)` against `(1 | h) + (1 | k)`). " *
        "Call `lrtest(reduced, full)` for the plain test, or use a parametric bootstrap."))
end

# A fit stranded on the failed-objective sentinel plateau (loglik <= -1e15, see
# `_sentinel_loglik`) has a meaningless likelihood: an LR statistic built from it is
# ~1e18 and its p-value 0 or 1, an AIC ~2e18. Comparison verbs refuse it, like the
# REML/VA refusals; information criteria return NaN with a warning.
function _sentinel_compare_guard(reduced::DrmFit, full::DrmFit, verb::AbstractString)
    for (nm, f) in (("reduced", reduced), ("full", full))
        _sentinel_loglik(f) && throw(ArgumentError(
            "$verb: the `$nm` fit is degenerate (loglik = $(f.loglik), the failed-" *
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

# Nested family pairs whose extra distributional parameter is tested at 0.
# Only the filed t10 pair is wired: Poisson ⊂ NegBinomial2 (sigma → 0).
# Sibling nests stay unwired; see the FX6 methods note.
function _family_boundary_nest(reduced_fam, full_fam)
    (reduced_fam isa Poisson && full_fam isa NegBinomial2) || return nothing
    return (added = :sigma, q = 1)
end
_family_boundary_nest(reduced::DrmFit, full::DrmFit) =
    _family_boundary_nest(reduced.family, full.family)

# Boundary caveat (issue #304, family-dispersion #1115): when `full` carries a
# variance-component block that `reduced` does not, OR when the pair is Poisson
# ⊂ NegBinomial2, the naive χ²(Δdof) LR reference is INVALID — the tested
# parameter sits on the boundary of its space, so the statistic follows a
# chi-bar-square mixture and the χ² p-value is CONSERVATIVE (too large; the test
# loses power). We do NOT silently swap the reference (the mixture order and
# independence assumptions need the caller's knowledge of the design), but we WARN
# and point to the boundary-corrected test. Silent otherwise.
function _boundary_vc_warn(reduced::DrmFit, full::DrmFit, verb::AbstractString)
    dropped = setdiff(_variance_component_blocks(full),
                      _variance_component_blocks(reduced))
    if !isempty(dropped)
        @warn(
            "$verb: `full` adds variance-component block(s) $(dropped) that `reduced` " *
            "lacks, so this compares a variance component against 0 — a BOUNDARY null. " *
            "The reported χ²($(dof(full) - dof(reduced))) p-value is INVALID there " *
            "(conservative: too large, so the test loses power); the correct reference " *
            "is a chi-bar-square mixture. Use `lrt_boundary(full, reduced; q = <#components>)` " *
            "(or a parametric bootstrap) for a boundary-correct p-value.")
        return nothing
    end
    nest = _family_boundary_nest(reduced, full)
    nest === nothing && return nothing
    @warn(
        "$verb: `full` is NegBinomial2 and `reduced` is Poisson, so this tests the extra " *
        "sigma at 0 — a BOUNDARY null. The reported χ²($(dof(full) - dof(reduced))) " *
        "p-value is INVALID there (conservative: too large, so the test loses power); " *
        "the correct reference is a chi-bar-square mixture. Use " *
        "`lrt_boundary(full, reduced; q = $(nest.q))` (or a parametric bootstrap) " *
        "for a boundary-correct p-value.")
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
function _map_compare_guard(a::DrmFit, b::DrmFit, verb::AbstractString)
    (a.estim_method === :MAP || b.estim_method === :MAP) || return nothing
    throw(ArgumentError(
        "$verb: cannot compare penalized (MAP) fits — priors or penalties alter the objective, " *
        "so the likelihood-ratio statistic does not have " *
        "the usual chi-square distribution. Refit both without priors or penalties to test, or compare " *
        "the penalized fits on their own terms."))
end

# Rows that entered the likelihood (#1053). Finite observed values are the
# storage-invariant fingerprint: a Gaussian `(1 | g)` fit listwise-drops a
# missing y, while the fixed-effect route keeps a NaN on the original length.
# When both fits store the same length, the finite-observation mask is compared
# too, so two padded fits that drop different rows still disagree even if the
# remaining y values coincide. `update` of a missing-response or imputed fit
# can refit a different set of rows than the seed (#1002).
function _obs_mask_vals(y::AbstractVector)
    mask = BitVector(undef, length(y))
    vals = Float64[]
    sizehint!(vals, length(y))
    @inbounds for (i, yi) in enumerate(y)
        fin = isfinite(yi)
        mask[i] = fin
        fin && push!(vals, yi)
    end
    return mask, vals
end

function _same_obs_rowset(mask_a, vals_a, mask_b, vals_b)
    vals_a == vals_b || return false
    length(mask_a) == length(mask_b) || return true
    return mask_a == mask_b
end

function _same_likelihood_rows(a::DrmFit, b::DrmFit)
    nobs(a) == nobs(b) || return false
    ka = sort!(collect(keys(a.obs)))
    kb = sort!(collect(keys(b.obs)))
    ka == kb || return false
    for k in ka
        ma, va = _obs_mask_vals(a.obs[k])
        mb, vb = _obs_mask_vals(b.obs[k])
        _same_obs_rowset(ma, va, mb, vb) || return false
    end
    return true
end

function _nobs_compare_guard(a::DrmFit, b::DrmFit, verb::AbstractString)
    _same_likelihood_rows(a, b) && return nothing
    detail = nobs(a) == nobs(b) ?
        "nobs = $(nobs(a)) on both, but the observation rows differ" :
        "nobs = $(nobs(a)) vs $(nobs(b))"
    throw(ArgumentError(
        "$verb: the two fits were estimated on different samples " *
        "($detail). A likelihood-ratio comparison " *
        "requires the same observations, which a missing-response or imputed " *
        "`update` does not guarantee. Refit both on the same rows."))
end

# Convergence caveat (#1002): a finite log-likelihood from a fit whose optimiser
# reported `converged = false` is still compared (the sentinel guard refuses the
# plateau), but its LR statistic can be off in either direction. Warn, naming the
# fit. A boundary fit (a variance pinned at its floor) has its own status from
# `convergence_status` and is not "unconverged", so it is skipped here.
function _unconverged_compare_warn(reduced::DrmFit, full::DrmFit, verb::AbstractString)
    bad = String[nm for (nm, f) in (("reduced", reduced), ("full", full))
                 if !f.converged && convergence_status(f) !== :boundary]
    isempty(bad) && return nothing
    @warn "$verb: the $(join(bad, " and ")) fit did not converge (`converged = false`). " *
          "The likelihood-ratio statistic may be wrong in either direction; refit " *
          "(different start values, optimiser, or tolerance) before relying on the p-value."
    return nothing
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
# an ELBO, not a log-likelihood; `:LA` (the route-specific default integrator) and
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
            " (`:LA` uses the route-specific default integrator; `:Laplace` is the one-point Laplace approximation)" : ""
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
    weights(fit::DrmFit) -> Union{Vector{Float64}, Nothing}

Prior (per-observation) weights the fit was made with, from
`drm(...; weights = w)`, or `nothing` when no weights were given. Extends
`StatsAPI.weights`.

The weights multiply each row's log-likelihood (drmTMB semantics), so
`loglik`, `aic`, `bic`, profiles and likelihood-ratio tests use the weighted
likelihood. `nobs(fit)` still counts rows, zero-weight rows included.

Prior weights are implemented on the fixed-effect ML routes of the univariate
families, on Gaussian and Poisson with one `(1 | g)`, on Gaussian `meta_V(v)`,
and on the residual bivariate Gaussian (see `test/test_weights_re_parity.jl`).
Any other route refuses `weights =` with an `ArgumentError` (#931).

Before #931 step (b), this accessor returned `ones(nobs(fit))` for every fit,
which looked like weights had been used.

# Example
```julia
fit  = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data)
weights(fit) === nothing          # no prior weights given
fitw = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data, weights = w)
weights(fitw) == w
```
"""
weights(fit::DrmFit) = fit.control_info === nothing ? nothing : fit.control_info.weights

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

Prior `weights`: if the seed fit has weights, `update` requires an explicit `weights =`
keyword and passes it to `drm`. Use `weights = weights(fit)` with the same `data`; use
`weights = nothing` to refit unweighted on purpose. If rows with a missing response were
dropped from the seed fit, `weights(fit)` has one entry per kept row and is shorter than
`data`, so pass the original full-length weight vector (one per row of `data`) instead.

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
    # A fit made with prior `weights` must not be refitted unweighted by accident. The fit
    # stores its weights but `update` takes new `data`, whose rows may differ, so the caller
    # says which weights to use: `weights = weights(fit)` for the same rows, new weights for
    # new rows, or `weights = nothing` to drop them on purpose.
    haskey(kwargs, :weights) || weights(fit) === nothing || throw(ArgumentError(
        "update: the fit was made with prior `weights`; pass `weights = weights(fit)` (same rows), " *
        "new weights for the new `data`, or `weights = nothing` to refit unweighted on purpose. " *
        "It is never refitted unweighted silently. If rows with a missing response were dropped from " *
        "the fit, `weights(fit)` has one entry per KEPT row and is shorter than `data`: pass the " *
        "original full-length weight vector (one per row of `data`) instead."))
    stored = _update_stored_options(fit)
    # A caller who changes the estimator does not want the old MAP penalty.
    # Passing `penalty` explicitly still overrides that drop.
    caller_method = get(kwargs, :method, nothing)
    if caller_method !== nothing && caller_method != fit.estim_method &&
            haskey(stored, :penalty) && !haskey(kwargs, :penalty)
        stored = Base.structdiff(stored, NamedTuple{(:penalty,)})
    end
    if caller_method !== nothing && caller_method != fit.estim_method &&
            haskey(stored, :priors) && !haskey(kwargs, :priors)
        stored = Base.structdiff(stored, NamedTuple{(:priors,)})
    end
    merged = merge(stored, NamedTuple(kwargs))
    refit = drm(formula, fit.family; data = data, merged...)
    # Replay the resolved specifications, while retaining the original table for
    # provenance. Do not reinterpret a broad selector against the new formula.
    if _is_fixed_prior_fit(fit) && haskey(fit.penalty, :input_table) &&
            haskey(merged, :priors) && !haskey(kwargs, :priors)
        return _with_prior_table_input(refit, fit.penalty.input_table)
    end
    return refit
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
    if _is_fixed_prior_fit(fit)
        push!(kw, :priors => fit.penalty.specs)
    elseif fit.penalty !== nothing
        push!(kw, :penalty => fit.penalty)
    end
    fit.marginal === :LA || push!(kw, :marginal => fit.marginal)
    return NamedTuple(kw)
end
