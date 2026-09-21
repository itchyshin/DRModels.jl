# missing_data.jl — listwise-deletion (complete-case) preprocessing for `drm`.
#
# Status (issue #49): DRModels.jl has NO native missing-data handling. A `missing` or
# `NaN` in a response or predictor column makes `drm(...)` ERROR (a `KeyError`
# from the StatsModels schema for `missing`, or `ArgumentError: matrix contains
# Infs or NaNs` from the linear-algebra solve for `NaN`) — it does not silently
# corrupt the fit, but the error is opaque about the cause. This file adds the
# smallest safe, documented path: explicit listwise deletion with a clear
# warning, so users have a guided complete-case route instead of hand-rolling row
# drops. It touches NO engine code — it is pure data preprocessing.
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

Why this exists: DRModels.jl has no native missing-data path, so a raw `missing`/`NaN`
response or predictor makes `drm` error with an opaque message. `drm_listwise`
gives a guided, documented complete-case route and emits a clear `@warn` reporting
how many rows (and which columns) were removed.

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
