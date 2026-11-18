# CumulativeLogit prediction design (#1121).
#
# The ordinal fitter drops the location intercept after `_design`, because the
# cutpoints absorb it (src/cumulative.jl), and the fit keeps the user's formula.
# The generic `_predict_design` (src/gaussian_core.jl) rebuilt the `mu` design
# from that formula, so StatsModels' implicit intercept came back and the #1000
# column check refused every `newdata`, the training data included, with a
# message about factor levels. This method makes the fitter's projection: build
# the design with the intercept, so factor contrasts are coded as at fit time,
# then drop that column. The #1000 check then runs on the projected names. η is
# the location linear predictor X·β, which drmTMB returns on both the link and
# the response scale (`_mean_response` is the identity for this family).
function _predict_design(fit::DrmFit{<:CumulativeLogit}, resp::Symbol, rhs, ndr, p::Symbol,
                         cache, off)
    _, X, nms = _design(resp, rhs, ndr; schema_cache = cache, schema_key = p,
                        store_schema = false)
    ic = findfirst(==("(Intercept)"), nms)
    if p === :mu && ic !== nothing
        keep = setdiff(1:length(nms), ic)
        X = X[:, keep]; nms = nms[keep]
    end
    fitted = _block_coefnames(fit, p)
    if fitted !== nothing && nms != fitted
        throw(ArgumentError("predict: the `$p` design built from `newdata` has columns " *
            "[" * join(nms, ", ") * "] but the fitted coefficients are [" *
            join(fitted, ", ") * "]. A factor in `newdata` probably has a different set " *
            "or order of levels than the data the model was fitted to; supply every " *
            "fitted level (unused rows can be dropped from the result afterwards)."))
    end
    η = X * coef(fit, p)
    off === nothing || (η = η .+ off)
    return X, η
end
