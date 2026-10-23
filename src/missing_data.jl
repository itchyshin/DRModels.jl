# missing_data.jl — listwise-deletion (complete-case) preprocessing for `drm`.
#
# Status (issues #49, #1089): `drm` drops the rows with a `missing` predictor,
# and a missing response on routes with no observed-rows path, with a warning
# that gives the count (R's `na.omit`; the helpers are at the end of this
# section). A `NaN` predictor still errors from the linear-algebra solve
# (#1113). `drm_listwise` below is the explicit complete-case route: it also
# drops `NaN`/`Inf` cells, and it is how to silence the warning. Neither
# touches engine code — both are pure data preprocessing.
#
# FIML for missing *responses* (the principled alternative that keeps partially
# observed rows — important for the bivariate / q=4 coevolution models) and
# multiple imputation for missing *predictors* are explicitly OUT OF SCOPE here
# and tracked as follow-up under issue #49 (see report/fiml-missing-data-design.md).

using Tables: Tables
using StatsModels: StatsModels

# True when a single cell should trigger dropping its row: `missing`, or a
# non-finite real (`NaN`/`±Inf`). Non-numeric, non-missing values never trigger.
_is_missing_cell(v) = ismissing(v) || (v isa Real && !isfinite(v))

# Response + predictor column symbols referenced by a (bivariate) formula bundle.
# Predictors come from `StatsModels.termvars` over each parameter's RHS; responses
# are the bundle's response column(s).
function _model_columns(f::DrmFormula)
    cols = Symbol[f.response]
    f.response2 === nothing || push!(cols, f.response2)
    for (_, rhs) in f.forms
        append!(cols, StatsModels.termvars(rhs))
    end
    return unique(cols)
end

function _model_columns(f::BivariateDrmFormula)
    cols = Symbol[f.response1, f.response2]
    for (_, rhs) in f.forms
        append!(cols, StatsModels.termvars(rhs))
    end
    return unique(cols)
end

# ---------------------------------------------------------------------------
# Row dropping inside `drm` (#1089). drmTMB drops every row with an NA in a
# model variable (R's `na.omit`), so `drm` does the same for a `missing`
# predictor and, on the routes with no observed-rows path of their own, for a
# missing response, with a warning that gives the count. `NaN`/`Inf` predictors
# are not dropped here: they are a different input error (#1113).

_formula_responses(f::DrmFormula) = (f.response, f.response2)
_formula_responses(f::BivariateDrmFormula) = (f.response1, f.response2)

# Columns a formula reads as predictors: every fixed-effect term of every
# distributional parameter, plus the covariate of a random slope `(x | g)`.
# Grouping columns and structured markers are left out — their columns are
# matched to a tree, a matrix or a time index, not read as numbers.
function _predictor_columns(f::Union{DrmFormula,BivariateDrmFormula})
    cols = Symbol[]
    for (_, rhs) in f.forms
        for t in (rhs isa Tuple ? rhs : (rhs,))
            if t isa FunctionTerm && t.f === (|)
                lhs = t.args[1]     # `(1 | tag | g)`: the inner `|` names a tag, not a column
                while lhs isa FunctionTerm && lhs.f === (|)
                    lhs = lhs.args[1]
                end
                append!(cols, StatsModels.termvars(lhs))
            elseif !(t isa FunctionTerm && t.f in (meta_V, relmat, animal, phylo, spatial, temporal))
                append!(cols, StatsModels.termvars(t))
            end
        end
    end
    return setdiff(unique(cols), _formula_responses(f))
end

# Subset every column to `keep`. A `Union{Missing,T}` vector that no longer
# holds a `missing` is narrowed to `T`, so a numeric predictor reaches the
# design as numbers; StatsModels would otherwise code it as a factor (#1111).
# Anything else (a `Vector{Any}` R factor from the bridge, a column that still
# holds a `missing`, a non-`Vector` array) keeps its own type.
function _subset_kept_rows(col, keep)
    sub = col[keep]
    S = eltype(sub)
    (sub isa Vector && S isa Union && Missing <: S && !any(ismissing, sub)) || return sub
    return convert(Vector{nonmissingtype(S)}, sub)
end

_subset_rows_narrowed(cols::NamedTuple, keep) =
    NamedTuple{keys(cols)}(map(col -> _subset_kept_rows(col, keep), values(cols)))

# `relmat`, `animal` and `spatial` line their matrix (or coordinates) up with
# the grouping levels in order of first appearance. Dropping rows is safe only
# when that level sequence survives; otherwise the matrix would silently pair
# with the wrong levels, so refuse by name. `phylo` matches by tip label.
function _check_drop_keeps_levels(f::Union{DrmFormula,BivariateDrmFormula}, cols::NamedTuple, keep)
    for (_, rhs) in f.forms, (kind, grp) in _collect_structured(rhs)
        kind === :phylo && continue
        haskey(cols, grp) || continue
        labels = cols[grp]
        isequal(unique(labels[keep]), unique(labels)) && continue
        throw(ArgumentError("drm: dropping the $(count(!, keep)) rows with a missing value " *
            "would change the levels of `$grp`, or their order of first appearance, and " *
            "`$kind(1 | $grp)` matches its matrix to those levels by position. Drop the rows " *
            "yourself (`drm_listwise`) and supply a matrix for the levels that remain."))
    end
    return nothing
end

# Prior `weights` (#931) give one value per row of `data`, so the rows dropped
# here take their weights with them (as in R, where `na.omit` subsets the
# model frame, weights included). A wrong-length vector is left for
# `_ctl_weights` to refuse.
function _drop_scope_weight_rows!(keep)
    s = _active_control_scope()
    (s === nothing || s.weights === nothing || length(s.weights) != length(keep)) && return nothing
    s.weights = s.weights[keep]
    return nothing
end

# Drop the rows with a `missing` predictor value (warning with the count) and
# return the narrowed column table; return `data` itself when nothing is
# dropped. An `mi(x)` joint formula models its missing predictors instead.
function _drop_missing_predictor_rows(f::Union{DrmFormula,BivariateDrmFormula}, data)
    any(p -> _joint_contains_mi(p.second), f.forms) && return data
    Tables.istable(data) || return data
    cols = Tables.columntable(data)
    pcols = filter(c -> haskey(cols, c) && Missing <: eltype(cols[c]), _predictor_columns(f))
    isempty(pcols) && return data
    n = length(cols[pcols[1]])
    keep = trues(n)
    offenders = Symbol[]
    for c in pcols
        hit = false
        for (i, v) in enumerate(cols[c])
            ismissing(v) && (keep[i] = false; hit = true)
        end
        hit && push!(offenders, c)
    end
    ndrop = count(!, keep)
    ndrop == 0 && return data
    ndrop < n || throw(ArgumentError("drm: every row has a missing value in a predictor " *
        "($(join(offenders, ", "))); there is nothing left to fit."))
    _check_drop_keeps_levels(f, cols, keep)
    @warn "drm: dropped $ndrop of $n rows with a missing predictor value " *
          "($(join(offenders, ", "))); the fit uses the complete cases, like R's `na.omit`. " *
          "Use `drm_listwise` to drop them explicitly and silence this."
    _drop_scope_weight_rows!(keep)
    return _subset_rows_narrowed(cols, keep)
end

# Drop the rows with a missing/NaN response (warning with the count), for the
# Gaussian routes that have no observed-rows path of their own.
function _drop_missing_response_rows(f::DrmFormula, data)
    observed = _formula_response_observed_mask(f, data)
    all(observed) && return data
    n = length(observed)
    ndrop = count(!, observed)
    ndrop < n || throw(ArgumentError("drm: at least one response row must be observed"))
    cols = Tables.columntable(data)
    _check_drop_keeps_levels(f, cols, observed)
    @warn "drm: $ndrop of $n rows have a missing/NaN response and were dropped (the fit " *
          "uses the complete cases, like R's `na.omit`). Use `drm_listwise` to drop them " *
          "explicitly and silence this."
    _drop_scope_weight_rows!(observed)
    return _subset_rows_narrowed(cols, observed)
end

"""
    drm_listwise(formula, data; verbose = true) -> NamedTuple

Drop every row of `data` that has a `missing` or non-finite (`NaN`/`Inf`) value in
any response or predictor column referenced by `formula` (a [`bf`](@ref) bundle or
the bivariate keyword form), returning a column `NamedTuple` ready to pass straight
to [`drm`](@ref). This is **listwise deletion** (a.k.a. complete-case analysis).

```julia
clean = drm_listwise(bf(y ~ x, sigma ~ x), data)        # @warn names the dropped rows
fit   = drm(bf(y ~ x, sigma ~ x), Gaussian(); data = clean)
```

Why this exists: `drm` itself drops rows with a `missing` predictor (and a missing
response where the route has no observed-rows path), with a warning that gives the
count. `drm_listwise` does the same drop explicitly, also drops `NaN`/`Inf` cells,
and emits one `@warn` reporting how many rows (and which columns) were removed, so
the data you pass to `drm` are exactly the rows it fits.

!!! warning "Listwise deletion discards information"
    For the **bivariate / q=4 coevolution** models a row is dropped if *either*
    trait is missing — exactly the information that full-information maximum
likelihood (FIML) is designed to keep. Listwise deletion is unbiased only
under missing-completely-at-random (MCAR) and is generally less efficient than
FIML under missing-at-random (MAR). Native FIML for missing responses and
multiple imputation for missing predictors are not currently available; this
helper is an interim complete-case path, not a substitute.

Only the columns the model uses are checked, so unrelated columns with missing
values do not cause rows to be dropped. Set `verbose = false` to suppress the
warning (the drop still happens). If no rows are dropped the input columns are
returned unchanged.
"""
function drm_listwise(formula::Union{DrmFormula,BivariateDrmFormula}, data; verbose::Bool = true)
    Tables.istable(data) ||
        throw(ArgumentError("drm_listwise: `data` must be a Tables.jl-compatible column table " *
                            "(e.g. a NamedTuple of vectors or a DataFrame)."))
    cols = Tables.columntable(data)
    used = _model_columns(formula)
    for c in used
        haskey(cols, c) ||
            throw(ArgumentError("drm_listwise: column `$c` referenced by the formula is not in `data`."))
    end

    n = length(first(cols))
    # A row is kept iff every model column is present + finite in that row.
    keep = trues(n)
    offenders = Symbol[]
    for c in used
        col = cols[c]
        bad = false
        @inbounds for i in 1:n
            if _is_missing_cell(col[i])
                keep[i] = false
                bad = true
            end
        end
        bad && push!(offenders, c)
    end

    ndropped = n - count(keep)
    if ndropped > 0 && verbose
        offenders_str = join(offenders, ", ")
        @warn "drm_listwise: dropped $ndropped of $n rows with missing/non-finite values " *
              "(listwise deletion); affected columns: $offenders_str. " *
              "This discards partial information — see issue #49 for the FIML alternative."
    end

    ndropped == 0 && return cols
    # Subset every column of the table (not just model columns) so downstream code
    # that reaches for extra columns still sees aligned, complete rows. `identity.`
    # re-narrows each kept column's eltype — dropping the now-impossible `Missing`
    # from a `Union{Missing,T}` response so the result is a plain `Vector{T}` that
    # `drm` accepts (an un-narrowed `Union{Missing,…}` column would still break the
    # fitter's numeric path).
    return NamedTuple{keys(cols)}(map(col -> identity.(col[keep]), values(cols)))
end
