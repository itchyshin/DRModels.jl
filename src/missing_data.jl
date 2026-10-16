# missing_data.jl — listwise-deletion (complete-case) preprocessing for `drm`.
#
# Status (issues #49 / #1089 / #1111): a `missing` in a *predictor* is refused
# by name (`drm_listwise`). A complete `Union{Missing,T<:Number}` predictor is
# narrowed so StatsModels keeps it numeric. `NaN`/`Inf` still error from the
# linear-algebra path. This file is the documented complete-case helper.
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

Why this exists: a raw `missing` in a **predictor** is refused by name (issue
#1089) rather than dropped inside `drm`. `drm_listwise` is the documented
complete-case route; it emits a clear `@warn` reporting how many rows (and which
columns) were removed. Incomplete **responses** are handled separately by the
observed-rows path inside `drm`.

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
warning (the drop still happens). After a complete-case pass, a
`Union{Missing,T<:Number}` predictor with no remaining missings is narrowed
to `T` so `drm` does not dummy-code it (issue #1111). `Vector{Any}` columns
(R `factor()` via the bridge) are left alone.
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

    if ndropped == 0
        # Even when nothing is dropped, narrow complete Union{Missing,T<:Number}
        # columns (#1111). Do not `identity.` a `Vector{Any}` R-factor column.
        names = keys(cols)
        return NamedTuple{names}(map(_narrow_union_missing_numeric, values(cols)))
    end
    # Subset every column of the table (not just model columns) so downstream code
    # that reaches for extra columns still sees aligned, complete rows. `identity.`
    # re-narrows each kept column's eltype — dropping the now-impossible `Missing`
    # from a `Union{Missing,T}` column so the result is a plain `Vector{T}`.
    return NamedTuple{keys(cols)}(map(col -> identity.(col[keep]), values(cols)))
end
