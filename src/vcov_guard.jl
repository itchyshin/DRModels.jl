# vcov_guard.jl — one guarded Hessian→covariance step for every fitter.
#
# Every family fitter used to end with a bare `V = inv(ForwardDiff.hessian(nll, θ̂))`. That is
# unsafe whenever the MLE sits on a variance/dispersion boundary, where the
# corresponding coordinate is unidentified and the Hessian is singular *by
# construction* — e.g. fitting NB2 to Poisson-generated data drives the size
# parameter to r → ∞, leaving that row/column numerically zero.
#
# Relying on `inv` to throw is not enough, and this is the bug it caused: on
# 2026-08-14 the `anchor (c)` VA test (test/test_variational.jl:105) threw
# SingularException(3) on CI Julia 1.10.11 and passed on 1.10.0 locally and on
# 1.11 in CI — same seeded data, same code. At the optimum θ̂[3] = −19.06
# (r = 3.6e16), diag(H)[3] = 9.9e-16 and cond(H) = 9.1e16: exactly the knife
# edge where LAPACK either reports a zero pivot or returns a finite but
# meaningless inverse. A try/inv/catch/pinv guard makes the *crash* go away but
# keeps the outcome platform-dependent and, worse, silently reports SE ≈ 0 for a
# parameter that is not estimable.
#
# So decide on the conditioning explicitly, identically on every platform, and
# say so out loud when the fit is at a boundary.

"""
Relative tolerance below which a Hessian eigenvalue counts as numerically zero.
"""
# CALIBRATED 2026-08-26 against the measured distribution, not chosen by feel.
# Was 1e-12, which only caught Hessians singular to MACHINE PRECISION and missed
# merely catastrophic ones -- a fit with a variance of 1.5e+07 in one direction
# reported an SE of ~3937 with no warning at all.
#
# Instrumented every _vcov_from_hessian call in the full suite (1,970 calls) on
# Linux, under both inner-solve tolerances. The distribution is BIMODAL below
# ~1e-7 and a smooth continuum of ordinary condition numbers above it:
#
#   arm            non-singular degenerate cluster    next point    empty gap
#   newton_tol 1e-8    9.84e-09 .. 1.62e-08           1.09e-07      0.83 orders
#   newton_tol 1e-10   4.14e-12 .. 5.53e-09           1.09e-07      1.30 orders
#
# 3e-8 splits that confirmed-empty gap: +0.73 orders above the degenerate cluster
# and +0.56 below the nearest other point. ZERO healthy fits are newly warned on
# in either arm -- the 38 calls that newly fire are 23 fixtures across three
# sparse-Laplace routes, every one with an implied SE of 2,192-6,136.
#
# NOTE this threshold is a fix for a specific hole, NOT a general boundary
# detector: above ~1e-7 the distribution is continuous, so no relative-ratio
# threshold up there could be principled.
#
# ALSO NOTE the hole is PRE-EXISTING, not introduced by tightening newton_tol:
# 6 calls already sit in the missed zone at the OLD tolerance. Tightening the
# inner solve widened it from 6 to 38 by moving fits off exact float zero onto
# the solver's own noise floor -- it made a latent weakness visible.
#
# Falsified by: a genuinely healthy fixture measuring <= ~1e-7; or this cluster's
# top (5.53e-9) creeping toward 3e-8 on another BLAS/LAPACK build -- the margin
# is 0.73 orders, comfortable but not enormous.
const _VCOV_RTOL = 3e-8

"""
    _vcov_from_hessian(H; context = "")

Covariance matrix from an observed-information (Hessian) matrix, guarded against
boundary degeneracy.

Symmetrises `H`, then inverts it, unless it is numerically singular or not
positive definite. A clearly negative eigenvalue means the point is not a
minimum: the function warns and returns a `NaN` covariance rather than
`inv(H)` or a pseudo-inverse. A numerically zero eigenvalue (a variance
boundary) warns, naming the flat coordinates; those flagged coordinates are
dropped from the Hessian before the rest is inverted, and their rows and columns of
the result are `NaN`. If no single coordinate is flagged (the singular direction is
a combination of several), the Moore-Penrose pseudo-inverse is used as before. The
decision is made from the eigenvalues rather than from whether `inv` happens to throw,
so the result does not depend on the LAPACK build or CPU.

Standard errors for flagged coordinates are withheld (`NaN`) because at a variance
boundary the sampling distribution is not the usual normal approximation. Use
`method = :profile` or the bootstrap entry points for those targets (see
`src/inference.jl`), or the χ̄² boundary machinery in `src/chibar.jl`.

Called on a bare matrix, the guard compares each curvature with the largest one, which
treats every coordinate as dimensionless. Called while a `drm` fit is being built, it
records the flag, and the `drm` entry re-judges it once the coordinates are named: a
coefficient on a covariate (or an identity-link location coefficient) is not flagged
for being small next to another coordinate, because its curvature depends on the units
of the data (see `_flat_units_free`). That step can only remove flags.
"""
function _vcov_from_hessian(H::AbstractMatrix; context::AbstractString = "")
    Hs = Matrix(H)
    Hs = Matrix(Symmetric((Hs + Hs') / 2))
    isempty(Hs) && return Hs

    # #1220: `eigvals` on a non-finite matrix raises LAPACK's bare "matrix contains
    # Infs or NaNs". Throw the same exception type (callers and owner decision 14
    # rely on it) but say what went wrong and what the user can do.
    if !all(isfinite, Hs)
        bad = sort!(unique!([i[1] for i in findall(!isfinite, Hs)]))
        ctx = isempty(context) ? "" : " ($(context))"
        throw(ArgumentError(
            "The Hessian of the negative log-likelihood at the optimum$(ctx) contains " *
            "Inf or NaN entries (parameter indices $(bad)), so the covariance matrix and " *
            "Wald standard errors cannot be computed. Likely causes: missing or non-finite " *
            "values (NaN/Inf) in the response or predictors, or a fit whose maximum " *
            "likelihood estimate does not exist, e.g. (quasi-)complete separation in a " *
            "binomial-type response, where a coefficient runs off towards ±Inf. What to do: " *
            "check the data for NaN/Inf; check whether a predictor perfectly separates " *
            "successes from failures (tabulate the response against it); simplify the model " *
            "or rescale predictors. `se = false` skips this step, but point estimates from " *
            "such a fit are not meaningful."))
    end

    ev = eigvals(Symmetric(Hs))
    scale = maximum(abs, ev)
    # A negative eigenvalue is a saddle or a ridge, not a variance boundary.
    # `minimum(abs, ev)` cannot see it, and `inv` of that matrix has a negative
    # diagonal (DRModels.jl#972). Do not pseudo-invert it either.
    if scale != 0 && minimum(ev) < -_VCOV_RTOL * scale
        neg = findall(<( -_VCOV_RTOL * scale), ev)
        @warn """
              Hessian is not positive definite at the optimum. Wald standard errors are withheld.
              A negative eigenvalue means this point is not a minimum, so inv(H) is not a covariance.
              """ context negative_eigenvalue_indices = neg
        return fill(NaN, size(Hs))
    end

    flagged_here = Int[]
    warn_here = false
    rcond_here = 0.0
    if scale == 0 || minimum(abs, ev) <= _VCOV_RTOL * scale
        d = abs.(diag(Hs))
        dmax = maximum(d)
        flat = dmax == 0 ? collect(eachindex(d)) : findall(<=(_VCOV_RTOL * dmax), d)
        # #1110: an aliased design column explains the singularity and was already
        # warned about by name; do not also blame a variance boundary.
        warn_singular = !_design_rank_pending()
        rcond = scale == 0 ? 0.0 : minimum(abs, ev) / scale
        if isempty(flat)
            V0 = pinv(Hs)
            warn_singular && _warn_singular_hessian(context, flat, rcond)
        else
            # Drop the flagged coordinates BEFORE inverting, so finite-difference
            # noise on a flat coordinate (1e-8 of either sign) cannot contaminate the
            # covariances of the others, and report the flagged rows and columns as
            # NaN: a variance for a coordinate whose curvature cannot be told from zero
            # is not a number the data support (DRModels.jl#1383).
            V0 = _vcov_drop_flagged(Hs, flat)
            flagged_here = flat
            warn_here = warn_singular
            rcond_here = rcond
        end
    else
        V0 = inv(Hs)
    end

    V = Matrix(Symmetric((V0 + V0') / 2))
    isempty(flagged_here) || _note_guard_flag!(Hs, V, flagged_here, context, rcond_here, warn_here)
    return V
end

# Covariance of the unflagged block of `Hs`; flagged rows and columns are NaN.
# The block is inverted directly when it is well conditioned and by pseudo-inverse
# otherwise (the behaviour the whole matrix had before), without a second warning.
function _vcov_drop_flagged(Hs::AbstractMatrix, flat::AbstractVector{<:Integer})
    n = size(Hs, 1)
    V = fill(NaN, n, n)
    keep = setdiff(1:n, flat)
    isempty(keep) && return V
    Hk = Hs[keep, keep]
    ev = eigvals(Symmetric(Hk))
    scale = maximum(abs, ev)
    V[keep, keep] = (scale != 0 && minimum(abs, ev) > _VCOV_RTOL * scale) ? inv(Hk) : pinv(Hk)
    return V
end

# Coordinates whose variance in `V` is NaN: the ones the guard flagged (#1383).
_flagged_coords(V::AbstractMatrix) = [i for i in axes(V, 1) if isnan(V[i, i])]

# ---------------------------------------------------------------------------
# Units-free judgement of the flag (#1383)
# ---------------------------------------------------------------------------
#
# `_vcov_from_hessian` sees only a matrix. It compares each diagonal curvature with
# the largest one, which is right when every coordinate is dimensionless (a log SD, a
# log size, an atanh correlation) but wrong for a coefficient on a covariate: its
# curvature scales with the squared units of that covariate. A covariate in units of
# 1e5 makes every other coordinate look flat, and one in units of 1e-5 makes its own
# coefficient look flat, in a well identified fit.
#
# A matrix alone cannot tell a flat coordinate from one in small units (the same
# diagonal in both cases), so the judgement is finished where the coordinates are
# named, once the fit exists: `_reconcile_guard_flags` (called by every `drm` entry)
# re-judges the flags with the criterion below. It can only remove flags.
#
#   * A coordinate that carries units (`_unit_bearing_coords`) is never flagged for being
#     small next to another coordinate.
#       - A coefficient on a covariate in a block with a log or logit link is judged per
#         unit of its own design column: its curvature divided by the squared norm of the
#         column (the mean per-observation weight) is flagged when it is at most
#         `_VCOV_RTOL`. Multiplying the column by c multiplies both by c^2, so the verdict
#         does not change.
#       - Any other such coordinate (a location coefficient of a Gaussian, Student or
#         skew-normal fit, whose weight carries the units of the response; an
#         off-diagonal covariance entry; a column that was not recorded) is flagged only
#         when its curvature is exactly zero.
#   * Every other coordinate (a log SD, a log size, a correlation, an intercept of a
#     log or logit block, a threshold) is flagged when its curvature is at most
#     `_VCOV_RTOL` times the number of observations. The reference does not come from
#     the other coordinates: a location intercept has curvature n / sigma^2 and a log
#     sigma has 2n, so comparing them would judge the signal to noise ratio, not
#     flatness. The curvature of a dimensionless coordinate that carries information
#     is of the order of the number of observations that inform it, and that does not
#     change with the units of any design column, with the units of the response, or
#     with the size of sigma. A flat coordinate is far below it.
#   * The result can only remove flags the matrix alone set: it is intersected with them.
#
# Known limits: weights or frequencies that are very small make every curvature small
# and are read as flatness; a flat slope in a block whose intercept is at the boundary
# is caught only when its design column was recorded.

const _GUARD_RECORDS_KEY = :drm_vcov_guard_records

const _GuardItem = NamedTuple{(:H, :V, :flat, :context, :rcond, :warn),
    Tuple{Matrix{Float64},Matrix{Float64},Vector{Int},String,Float64,Bool}}

mutable struct _GuardRecords
    lock::ReentrantLock
    items::Vector{_GuardItem}
    # Squared norm of each design column built during the call, by coefficient name
    # (the smallest when a name occurs more than once); see `_note_design_norms!`.
    norms::Dict{String,Float64}
end
_GuardRecords() = _GuardRecords(ReentrantLock(), _GuardItem[], Dict{String,Float64}())

# Called by `_design` for every design matrix it builds. Inside a `drm` call the squared
# norm of each column is kept, so that the coefficient on a covariate can be judged
# per unit of that covariate (`_flat_units_free`). Elsewhere it does nothing.
function _note_design_norms!(names, X::AbstractMatrix)
    r = get(task_local_storage(), _GUARD_RECORDS_KEY, nothing)
    r isa _GuardRecords || return nothing
    lock(r.lock) do
        for (j, nm) in enumerate(names)
            j <= size(X, 2) || break
            v = sum(abs2, view(X, :, j))
            key = String(nm)
            r.norms[key] = haskey(r.norms, key) ? min(r.norms[key], v) : v
        end
    end
    return nothing
end

function _warn_singular_hessian(context, flat, rcond)
    @warn """
          Hessian is numerically singular at the optimum — using a pseudo-inverse.
          The fit is at a variance/dispersion boundary, so standard errors for the
          flagged coordinates are NOT trustworthy. Prefer `confint(fit, method = :profile)`
          or the bootstrap entry points for those targets.
          """ context flat_coordinates = flat rcond
    return nothing
end

# Called by the guard when it flagged coordinates. Inside a `drm` call the record is
# kept and the warning is left to `_reconcile_guard_flags`, which knows the final flag;
# anywhere else the warning is given at once.
function _note_guard_flag!(H::AbstractMatrix, V::AbstractMatrix, flat::AbstractVector{<:Integer},
                           context, rcond::Real, warn::Bool)
    r = get(task_local_storage(), _GUARD_RECORDS_KEY, nothing)
    if !(r isa _GuardRecords)
        warn && _warn_singular_hessian(context, collect(Int, flat), rcond)
        return nothing
    end
    lock(r.lock) do
        push!(r.items, (H = Matrix{Float64}(H), V = Matrix{Float64}(V), flat = Vector{Int}(flat),
                        context = String(context), rcond = Float64(rcond), warn = warn))
    end
    return nothing
end

_identity_location_family(fam) = fam isa Gaussian || fam isa Student || fam isa SkewNormal

# Blocks whose non-intercept columns are design columns of a covariate.
const _COEF_BLOCKS = (:mu, :mu1, :mu2, :sigma, :sigma1, :sigma2, :nu, :zi, :hu, :zoi, :coi, :rho12)

# Off-diagonal entry of a Cholesky factor of a covariance (`g:L21`): in response units.
_is_offdiag_chol_name(nm::AbstractString) =
    (m = match(r":L(\d)(\d)$", nm)) !== nothing && m.captures[1] != m.captures[2]

"""
    _unit_bearing_coords(fit) -> BitVector

Coordinates of `fit.theta` whose curvature depends on the units of the data: every
coefficient of an identity-link location block (units of the response), every other
coefficient on a design column that is not the intercept, and an off-diagonal
covariance Cholesky entry. Everything else (intercepts of log and logit blocks, log
SDs, log sizes, correlations, thresholds) is dimensionless.
"""
function _unit_bearing_coords(fit)
    unit = falses(length(fit.theta))
    idloc = _identity_location_family(fit.family)
    for ((p, rng), (_, nms)) in zip(fit.blocks, fit.coefnames)
        for (k, i) in enumerate(rng)
            (1 <= i <= length(unit)) || continue
            nm = k <= length(nms) ? String(nms[k]) : ""
            if idloc && p in (:mu, :mu1, :mu2)
                unit[i] = true
            elseif p in _COEF_BLOCKS && nm != "(Intercept)"
                unit[i] = true
            elseif _is_offdiag_chol_name(nm)
                unit[i] = true
            end
        end
    end
    return unit
end

# Squared norm of the design column behind each coordinate of `fit.theta` that carries
# units but whose per-observation weight does not: a coefficient on a covariate in a
# block with a log or logit link (not an identity-link location block, whose weight
# carries the units of the response, and not a covariance Cholesky entry). `NaN` for
# every other coordinate, and for a column that was not recorded.
function _column_norms2(fit, recs::_GuardRecords, unit::AbstractVector{Bool})
    cn2 = fill(NaN, length(fit.theta))
    idloc = _identity_location_family(fit.family)
    for ((p, rng), (_, nms)) in zip(fit.blocks, fit.coefnames)
        (idloc && p in (:mu, :mu1, :mu2)) && continue
        for (k, i) in enumerate(rng)
            (1 <= i <= length(cn2) && unit[i] && k <= length(nms)) || continue
            _is_offdiag_chol_name(String(nms[k])) && continue
            cn2[i] = lock(() -> get(recs.norms, String(nms[k]), NaN), recs.lock)
        end
    end
    return cn2
end

# The flagged coordinates under the units-free criterion above. `colnorm2` (see
# `_column_norms2`) lets a coefficient on a covariate be judged per unit of its own
# column: its curvature divided by the squared norm of the column is the mean
# per-observation weight, a dimensionless number that does not change when the column
# is multiplied by a constant.
function _flat_units_free(H::AbstractMatrix, unit::AbstractVector{Bool}, nobs::Real;
                          colnorm2::Union{Nothing,AbstractVector{<:Real}} = nothing)
    d = abs.(diag(H))
    dmax = maximum(d)
    ref = Float64(nobs)
    flat = Int[]
    for i in eachindex(d)
        if !unit[i]
            d[i] <= _VCOV_RTOL * ref && push!(flat, i)
        elseif colnorm2 !== nothing && isfinite(colnorm2[i]) && colnorm2[i] > 0
            d[i] <= _VCOV_RTOL * colnorm2[i] && push!(flat, i)
        else
            (dmax == 0 || d[i] == 0) && push!(flat, i)
        end
    end
    return flat
end

# Inverse of a symmetric positive definite block, computed on its unit-diagonal form
# so that coordinates in very different units do not spoil the conditioning; the
# pseudo-inverse of that form (or of the block, when a diagonal entry is not positive)
# otherwise.
function _inverse_units_free(Hk::AbstractMatrix)
    d = diag(Hk)
    if all(>(0), d)
        s = 1 ./ sqrt.(d)
        C = Hk .* s .* s'
        ev = eigvals(Symmetric(C))
        if minimum(ev) > _VCOV_RTOL * maximum(ev)
            return Matrix(Symmetric(inv(Symmetric(C)) .* s .* s'))
        end
        return Matrix(Symmetric(pinv(Symmetric(C)) .* s .* s'))
    end
    return pinv(Hk)
end

function _vcov_units_free(H::AbstractMatrix, flat::AbstractVector{<:Integer})
    n = size(H, 1)
    V = fill(NaN, n, n)
    keep = setdiff(1:n, flat)
    isempty(keep) && return V
    V[keep, keep] = _inverse_units_free(H[keep, keep])
    return Matrix(Symmetric((V + V') / 2))
end

# Finish the flag of a finished fit (see the comment above). `recs` holds what the
# guard flagged while the fit was being built. The covariance of a fit with an aliased
# design column was rewritten after the guard (`_apply_design_rank` sets the aliased
# rows and columns to NaN), so the record is matched ignoring those rows and columns,
# and they are set to NaN again. A covariance that is not one of the records (the route
# changed it in another way) is left alone.
function _reconcile_guard_flags(fit, recs::_GuardRecords)
    isempty(recs.items) && return fit
    alias = fit isa DrmFit ? _aliased_coords(fit) : Int[]
    k = fit isa DrmFit ?
        lock(() -> findlast(r -> _same_except(r.V, fit.vcov, alias), recs.items), recs.lock) : nothing
    rec = k === nothing ? nothing : recs.items[k]
    if rec !== nothing && size(rec.H, 1) != length(fit.theta)
        rec = nothing
    end
    # Warnings the guard deferred: the ones of covariances that were not kept as they are.
    for r in recs.items
        (r.warn && r !== rec) && _warn_singular_hessian(r.context, r.flat, r.rcond)
    end
    rec === nothing && return fit
    # The units-free flag can only remove flags the matrix alone set.
    unit = _unit_bearing_coords(fit)
    flat = intersect(_flat_units_free(rec.H, unit, fit.nobs; colnorm2 = _column_norms2(fit, recs, unit)),
                     rec.flat)
    rec.warn && !isempty(flat) && _warn_singular_hessian(rec.context, flat, rec.rcond)
    V = if flat == rec.flat
        fit.vcov
    else
        Vn = _vcov_units_free(rec.H, flat)
        for c in alias
            Vn[c, :] .= NaN
            Vn[:, c] .= NaN
        end
        Vn
    end
    # One stored flag, for the case the covariance cannot show: every coordinate flagged
    # gives an all-NaN matrix, which is otherwise the same as a failed covariance.
    uncertainty = (length(flat) == length(fit.theta) && fit.uncertainty === :ok) ?
        :boundary_all : fit.uncertainty
    (flat == rec.flat && uncertainty === fit.uncertainty) && return fit
    return _withvcov(fit, V, uncertainty)
end

# `a` and `b` agree (NaN equal to NaN) outside the rows and columns `skip`, and `b` is NaN
# on them.
function _same_except(a::AbstractMatrix, b::AbstractMatrix, skip::AbstractVector{<:Integer})
    isempty(skip) && return isequal(a, b)
    size(a) == size(b) || return false
    n = size(a, 1)
    for j in 1:n, i in 1:n
        if i in skip || j in skip
            isnan(b[i, j]) || return false
        elseif !isequal(a[i, j], b[i, j])
            return false
        end
    end
    return true
end

# Row-wise delta-method variances `xᵢ' V xᵢ` for the rows of `X`. A row that puts weight
# on a flagged coordinate gets NaN; a row that does not keeps its variance, because
# `0 * NaN` must not remove it. With no flagged coordinate this is the plain product.
function _delta_variances(X::AbstractMatrix, V::AbstractMatrix)
    flagged = _flagged_coords(V)
    isempty(flagged) && return vec(sum((X * V) .* X; dims = 2))
    keep = setdiff(axes(V, 1), flagged)
    Xk = X[:, keep]
    out = vec(sum((Xk * V[keep, keep]) .* Xk; dims = 2))
    for i in axes(X, 1)
        any(!iszero, view(X, i, flagged)) && (out[i] = NaN)
    end
    return out
end

"""
    _vcov_from_hessian_checked(H, data_finite; context = "") -> (V, hessian_finite)

[`_vcov_from_hessian`](@ref) for a ForwardDiff Hessian at an optimum whose
objective may have run off to a degenerate limit (#968, #961): a residual scale
collapsing towards 0 overflows the curvature to `Inf`/`NaN`. When the DATA are
finite that is a degenerate fit, not a NaN input, so it warns and returns an
all-`NaN` covariance plus `hessian_finite = false` (callers report
`converged = false`) instead of the LAPACK `ArgumentError` the guard raises on a
non-finite matrix. With non-finite data the guard is called unchanged and keeps
throwing (owner decision 14: NaN-predictor fits rely on it).
"""
function _vcov_from_hessian_checked(H::AbstractMatrix, data_finite::Bool;
                                    context::AbstractString = "")
    hf = all(isfinite, H)
    (hf || !data_finite) && return (_vcov_from_hessian(H; context = context), hf)
    @warn "The Hessian is not finite at the optimum (a scale parameter has run off " *
          "towards a degenerate limit, e.g. a constant or badly scaled response). Wald " *
          "standard errors are withheld and the fit is reported as not converged." context
    return (fill(NaN, size(H)), false)
end

"""
    _vcov_from_gram(G; context = "")

Inverse of a design Gram / GLS information block `X'WX`. Its eigenvalue ratio
reflects the units of the predictors (an uncentred `year ≈ 2000` gives ~1e-11),
which says nothing about identifiability, so a positive-definite Gram is inverted
directly and silently (as before the REML sites went through the guard, #956).
Only a matrix that is not numerically positive definite is passed to
[`_vcov_from_hessian`](@ref), which warns and falls back.
"""
function _vcov_from_gram(G::AbstractMatrix; context::AbstractString = "")
    Gs = Matrix(Symmetric((Matrix(G) + Matrix(G)') / 2))
    all(isfinite, Gs) || return _vcov_from_hessian(Gs; context = context)
    F = cholesky(Symmetric(Gs); check = false)
    return issuccess(F) ? Matrix(Symmetric(inv(F))) : _vcov_from_hessian(Gs; context = context)
end

_all_finite(xs...) = all(x -> all(isfinite, x), xs)

"""
    _fd_hessian_from_grad(grad_at, θ̂; hstep = 1e-6, retry_step = 1e-4) -> (H, ok)

Central finite-difference Hessian of an analytic gradient `grad_at(θ)`.
The sparse LSS `eval_core` / `_lss_sparse_multi_objective_and_grad` return an
EMPTY gradient when the nll or gradient is not finite at a probe point. Each
column is retried once with `retry_step`; if the retry also fails (empty,
wrong length, or non-finite), `ok = false` is returned instead of throwing a
`DimensionMismatch`. `scaled = false` uses `hstep` as an absolute step (no
`max(|θ̂ₖ|, 1)` scaling). Callers report the NaN-vcov convention when `!ok`.
"""
function _fd_hessian_from_grad(grad_at, θ̂::AbstractVector; hstep::Real = 1e-6,
                               retry_step::Real = 1e-4, scaled::Bool = true)
    np = length(θ̂)
    H = zeros(np, np)
    usable(g) = length(g) == np && all(isfinite, g)
    for k in 1:np
        gp = gm = Float64[]
        step = scaled ? hstep * max(abs(θ̂[k]), 1.0) : Float64(hstep)
        for (attempt, s) in enumerate((step, Float64(retry_step)))
            step = s
            θp = collect(Float64, θ̂); θm = collect(Float64, θ̂)
            θp[k] += step; θm[k] -= step
            gp = grad_at(θp); gm = grad_at(θm)
            usable(gp) && usable(gm) && break
        end
        (usable(gp) && usable(gm)) || return (H, false)
        H[:, k] .= (gp .- gm) ./ (2 * step)
    end
    H .= 0.5 .* (H .+ H')
    return (H, true)
end

"""
    _fd_hessian_from_values(f, θ̂; hstep = 1e-5, sentinel = 1e16) -> (H, ok)

Central finite-difference Hessian of an objective VALUE `f(θ)` (4-point mixed
differences, upper triangle mirrored). Value-based sibling of
[`_fd_hessian_from_grad`](@ref). Several objectives signal a failed evaluation
(non-PD factor, non-finite nll) with a finite `1e18` penalty rather than `NaN`;
differencing that sentinel yields a huge but finite garbage Hessian that then
passes the eigenvalue guard. Any probe that is non-finite or `≥ sentinel`
returns `ok = false` at once, and the caller reports the NaN-vcov convention.
"""
function _fd_hessian_from_values(f, θ̂::AbstractVector; hstep::Real = 1e-5,
                                 sentinel::Real = 1e16)
    np = length(θ̂)
    H = zeros(np, np)
    usable(v) = isfinite(v) && v < sentinel
    for k in 1:np, j in k:np
        sk = hstep * max(abs(θ̂[k]), 1.0)
        sj = hstep * max(abs(θ̂[j]), 1.0)
        θpp = collect(Float64, θ̂); θpm = copy(θpp); θmp = copy(θpp); θmm = copy(θpp)
        θpp[k] += sk; θpp[j] += sj
        θpm[k] += sk; θpm[j] -= sj
        θmp[k] -= sk; θmp[j] += sj
        θmm[k] -= sk; θmm[j] -= sj
        vpp, vpm, vmp, vmm = f(θpp), f(θpm), f(θmp), f(θmm)
        (usable(vpp) && usable(vpm) && usable(vmp) && usable(vmm)) || return (H, false)
        H[k, j] = (vpp - vpm - vmp + vmm) / (4 * sk * sj)
        H[j, k] = H[k, j]
    end
    return (H, true)
end

"""
    _vcov_from_fd_hessian(H; context = "")

[`_vcov_from_hessian`](@ref) for a Hessian from `_finite_hessian`, which signals a
failed stencil (non-finite probe or the failed-fit sentinel) with an all-NaN
matrix and a warning. That failure is reported as the same-shape all-NaN vcov (the
NaN-vcov convention of the other guarded Hessian helpers) rather than crashing in
`eigvals`. Deliberately NOT folded into `_vcov_from_hessian`: elsewhere a NaN
Hessian (e.g. NaN in a predictor) must keep throwing.
"""
function _vcov_from_fd_hessian(H::AbstractMatrix; context::AbstractString = "")
    all(isfinite, H) || return fill(NaN, size(H)...)
    return _vcov_from_hessian(H; context = context)
end
