# finite_inputs.jl — one front-door check for non-finite model inputs, and the
# converged-flag backstop that every DrmFit constructor applies.
#
# Responses, predictors, weights, offsets and coordinates used to reach a
# fitter and come back as a DrmFit with a non-finite log-likelihood or
# coefficient and `converged = true` (#1009 temporal, #1012 spatial coords,
# #1019 Student random intercept, #1021 associate_pairs). Every formula route
# builds its response and design through `_coerce_response_column` / `_design`,
# which call `_require_finite_inputs`. Routes that never build a design
# (raw-array mixed fits, the q2 bridge, `associate_pairs`, and every spatial
# distance builder) call the same function themselves.

"""
    _require_finite_inputs(; response=nothing, predictors=nothing,
                           weights=nothing, offset=nothing, coords=nothing,
                           response_allow_missing=true, context="drm")

Reject non-finite inputs before a fit or a staged association.

`response` is `(values, name)`. `missing` and `NaN` are the observed-rows
pattern and are allowed when `response_allow_missing` is true; `Inf` and
`-Inf` are never a missing value. `predictors` is `(matrix_or_column, names)`
and rejects every non-finite entry, naming the column. `weights`, `offset`
and `coords` are arrays; a non-finite entry names that argument and the first
offending index.

`context` is the leading word of the `ArgumentError` (`"drm"`,
`"associate_pairs"`, …).
"""
function _require_finite_inputs(; response=nothing, predictors=nothing,
                                weights=nothing, offset=nothing, coords=nothing,
                                response_allow_missing::Bool=true,
                                context::AbstractString="drm")
    response === nothing || _require_finite_response(response[1], response[2];
        allow_missing=response_allow_missing, context=context)
    predictors === nothing || _require_finite_predictors(predictors[1], predictors[2];
        context=context)
    weights === nothing || _require_finite_array(weights, "weights"; context=context)
    offset === nothing || _require_finite_array(offset, "offset"; context=context)
    coords === nothing || _require_finite_array(coords, "coords"; context=context)
    return nothing
end

# `missing` / `NaN` may mark an unobserved response. `Inf` may not.
function _input_offender(x; allow_missing::Bool)
    if x === missing
        return allow_missing ? nothing : "missing"
    end
    x isa AbstractFloat && isnan(x) && return allow_missing ? nothing : "NaN"
    x isa Real || return nothing
    xf = Float64(x)
    return isfinite(xf) ? nothing : string(xf)
end

function _require_finite_response(values, name; allow_missing::Bool, context::AbstractString)
    for (i, x) in pairs(values)
        bad = _input_offender(x; allow_missing=allow_missing)
        bad === nothing && continue
        hint = allow_missing ?
            " missing and NaN mark an unobserved response and are omitted; Inf is not a missing value." :
            ""
        throw(ArgumentError(
            "$context: response `$name` contains a non-finite value ($bad) at row $i.$hint"))
    end
    return nothing
end

function _require_finite_predictors(values, names; context::AbstractString)
    if values isa AbstractVector
        label = names isa AbstractString ? names : (isempty(names) ? "predictor" : string(first(names)))
        _require_finite_array(values, label; context=context, kind="predictor")
        return nothing
    end
    values isa AbstractMatrix || return nothing
    ncol = size(values, 2)
    for j in 1:ncol
        label = if names isa AbstractString
            ncol == 1 ? names : "$names column $j"
        elseif names isa AbstractVector && j <= length(names)
            string(names[j])
        else
            "column $j"
        end
        for i in 1:size(values, 1)
            bad = _input_offender(values[i, j]; allow_missing=false)
            bad === nothing && continue
            throw(ArgumentError(
                "$context: predictor `$label` contains a non-finite value ($bad) at row $i."))
        end
    end
    return nothing
end

function _require_finite_array(values, argument; context::AbstractString="drm", kind::AbstractString="")
    values isa AbstractArray || return nothing
    prefix = kind == "" ? "`$argument`" : "$kind `$argument`"
    if ndims(values) == 1
        for (i, x) in pairs(values)
            bad = _input_offender(x; allow_missing=false)
            bad === nothing && continue
            throw(ArgumentError(
                "$context: $prefix contains a non-finite value ($bad) at row $i."))
        end
        return nothing
    end
    for I in CartesianIndices(values)
        bad = _input_offender(values[I]; allow_missing=false)
        bad === nothing && continue
        loc = ndims(values) == 2 ? "row $(I[1]), column $(I[2])" : "index $I"
        throw(ArgumentError(
            "$context: $prefix contains a non-finite value ($bad) at $loc."))
    end
    return nothing
end

# Failed-fit sentinels. A flat objective is reported as nll = 1e18, so the
# stored log-likelihood is -1e18; `associate_pairs` used to store
# `-prevfloat(Inf)` (`-floatmax`). Both, and any non-finite log-likelihood,
# sit at or below -1e15. A genuine log-likelihood of -1e15 would need on the
# order of 1e14 observations. The same bar is what `is_converged` already uses.
function _loglik_is_sentinel(loglik)::Bool
    ll = try
        Float64(loglik)
    catch
        return true
    end
    return !isfinite(ll) || ll <= -1e15
end

function _coefficients_finite(theta)::Bool
    for x in theta
        isfinite(x) || return false
    end
    return true
end

"""
    _report_converged(converged, loglik, theta) -> Bool

Optimiser flag cleared when the stored log-likelihood is non-finite or a
failed-fit sentinel (`-1e18`, `-floatmax`, or any value `≤ -1e15`), or when
any coefficient is non-finite. Every `DrmFit` constructor uses this, and
[`is_converged`](@ref) agrees on the same two conditions.
"""
function _report_converged(converged, loglik, theta)::Bool
    converged || return false
    _loglik_is_sentinel(loglik) && return false
    _coefficients_finite(theta) || return false
    return true
end
