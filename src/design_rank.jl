# design_rank.jl: rank check of each parameter's fixed-effect design (#1110).
#
# A design with an aliased column (`sigma ~ 1 + x + z` with `z = 2x + 1`, or a
# duplicated column) leaves the likelihood flat along the alias direction. The
# optimiser still "converges", the Hessian is singular, and the guarded
# pseudo-inverse in `_vcov_from_hessian` reports finite SEs for coefficients
# that are not identified. drmTMB reports `is_converged` FALSE and a
# `fixed_effect_rank_deficiency` warning; this file does the same here.
#
# Scope: the check runs once per fit on the Gaussian location-scale route
# (`_drm_gaussian_fit`). `_check_fixed_design_rank` takes any list of
# (parameter, design, names), so another family's `drm` method can call the same
# hook. The pseudo-inverse itself is unchanged: only the aliased coordinates'
# covariance rows/columns are set to NaN, and the fit is flagged.

# Relative tolerance on a column's residual after projecting out the columns
# before it. An exact alias leaves ~1e-16; a merely near-collinear pair
# (r = 0.9999 gives ~1e-2) is far above this.
const _DESIGN_RANK_RTOL = 1e-10

"""
    _design_rank(X, names) -> Union{Nothing, NamedTuple}

`nothing` when `X` has full column rank. Otherwise `(rank, ncol, aliased)`:
`rank` comes from a column-pivoted QR (`|R[i,i]| > rtol * |R[1,1]|`) and
`aliased` names the columns that are linear combinations of the columns before
them, so the reported name is the later duplicate (`z`, not the intercept).
"""
function _design_rank(X::AbstractMatrix{<:Real}, names::AbstractVector{<:AbstractString})
    n, p = size(X)
    (p == 0 || n == 0 || !all(isfinite, X)) && return nothing
    R = abs.(diag(qr(Matrix{Float64}(X), ColumnNorm()).R))
    rk = count(>(_DESIGN_RANK_RTOL * R[1]), R)
    rk >= p && return nothing
    # Aliased names: modified Gram-Schmidt in column order.
    Q = Vector{Vector{Float64}}()
    aliased = String[]
    for j in 1:p
        v = Float64.(X[:, j]); nv = norm(v)
        for q in Q
            v .-= dot(q, v) .* q
        end
        if norm(v) <= _DESIGN_RANK_RTOL * nv || nv == 0
            push!(aliased, String(names[j]))
        else
            push!(Q, v ./ norm(v))
        end
    end
    # The two counts agree for any sane design; fall back to the pivot order if not.
    if length(aliased) != p - rk
        piv = qr(Matrix{Float64}(X), ColumnNorm()).p
        aliased = String.(names[sort(piv[rk+1:end])])
    end
    return (rank = rk, ncol = p, aliased = aliased)
end

# Per-fit findings travel from the design hook to the `drm` entry wrapper through
# task-local storage, so no fitter signature changes.
const _DESIGN_RANK_KEY = :drm_design_rank_findings

_design_rank_pending() = get(task_local_storage(), _DESIGN_RANK_KEY, nothing) !== nothing

function _design_rank_message(f)
    parts = ["`$(b.param)` block: $(join(b.aliased, ", ")) (design rank $(b.rank) of $(b.ncol) columns)"
             for b in f]
    return "fixed_effect_rank_deficiency: a fixed-effect design is rank-deficient, so some " *
           "coefficients are not identified. Aliased coefficient(s): " * join(parts, "; ") *
           ". Drop or recode the aliased column(s). Wald SEs for the aliased coefficient(s) are " *
           "reported NaN and the fit is flagged not converged (`is_converged`); the " *
           "log-likelihood and the other estimates are unaffected."
end

"""
    _check_fixed_design_rank(designs...)

Hook called once after the designs are built. Each argument is
`(param::Symbol, X, names)`. Records the rank-deficient blocks for the `drm`
entry wrapper (`_apply_design_rank`) and warns once. Clears any stale record
from an earlier fit on the same task first.
"""
function _check_fixed_design_rank(designs...)
    found = NamedTuple[]
    for (param, X, nms) in designs
        r = _design_rank(X, nms)
        r === nothing || push!(found, (param = param, r...))
    end
    if isempty(found)
        delete!(task_local_storage(), _DESIGN_RANK_KEY)
    else
        task_local_storage(_DESIGN_RANK_KEY, found)
        @warn _design_rank_message(found) aliased_blocks = found
    end
    return nothing
end

# fit -> findings. Keyed by the identity of `fit.theta`, which every `_with*`
# rebuilder of `DrmFit` passes through unchanged; a WeakRef guards against an
# `objectid` being reused after the fit is collected.
const _DESIGN_RANK_FITS = Dict{UInt,Tuple{WeakRef,Vector{NamedTuple}}}()
const _DESIGN_RANK_LOCK = ReentrantLock()

function _design_rank_findings(fit)
    lock(_DESIGN_RANK_LOCK) do
        e = get(_DESIGN_RANK_FITS, objectid(fit.theta), nothing)
        (e === nothing || e[1].value !== fit.theta) ? nothing : e[2]
    end
end

"""
    _design_rank_deficient(fit) -> Bool

Whether the fit was built from a rank-deficient fixed-effect design. Read by
`_nondegenerate_fit` (so `is_converged` is false) and by `check_drm`.
"""
_design_rank_deficient(fit) = _design_rank_findings(fit) !== nothing

"""
    _apply_design_rank(fit) -> DrmFit

Called by the `drm` entry on the finished fit. With no pending finding it
returns `fit` untouched. Otherwise it sets the covariance rows/columns of every
aliased coefficient to `NaN` (the pseudo-inverse of the rest is unchanged),
registers the fit as rank-deficient and clears the pending record.
"""
function _apply_design_rank(fit)
    found = get(task_local_storage(), _DESIGN_RANK_KEY, nothing)
    delete!(task_local_storage(), _DESIGN_RANK_KEY)
    (found === nothing || !(fit isa DrmFit)) && return fit
    V = copy(fit.vcov)
    for ((p, rng), (_, nms)) in zip(fit.blocks, fit.coefnames)
        i = findfirst(b -> b.param === p, found)
        i === nothing && continue
        for (k, nm) in enumerate(nms)
            if nm in found[i].aliased && k <= length(rng) && last(rng) <= size(V, 1)
                c = rng[k]
                V[c, :] .= NaN
                V[:, c] .= NaN
            end
        end
    end
    out = _withcontrolinfo(fit, V, fit.control_info)
    lock(_DESIGN_RANK_LOCK) do
        filter!(kv -> kv[2][1].value !== nothing, _DESIGN_RANK_FITS)
        _DESIGN_RANK_FITS[objectid(out.theta)] = (WeakRef(out.theta), found)
    end
    return out
end

# check_drm field: `nothing` unless the fit's design was rank-deficient.
function _fixed_effect_rank_deficiency(fit)
    f = _design_rank_findings(fit)
    f === nothing && return nothing
    return (deficient = true, blocks = f)
end
