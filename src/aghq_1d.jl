# aghq_1d.jl — DRModels-native 1-D Liu–Pierce adaptive Gauss–Hermite quadrature
# around existing `_gauss_hermite` (#448, lever 2).
#
# This is plumbing, not a recovery headline. k=1 recovers the 1-point Laplace
# approximation of ∫ exp(logf) — that identity proves the change of variables,
# not that quadrature is accurate. Tensor / multi-d AGHQ is intentionally
# absent: on tree-sized phylo Laplace it is a category error (k^{tree}).
#
# Do not vendor GLLVM `aghq_grid.jl` or drmTMB `R/aghq-coxreid.R` (GPL).
# Do not relabel the production `(1|g)` GHQ-32 `:LA` path as AGHQ.

"""
    _aghq_require_1d(dim)

Throw `ArgumentError` unless `dim == 1`. 1-D Liu–Pierce AGHQ only (#448).
"""
function _aghq_require_1d(dim::Integer)
    dim == 1 || throw(ArgumentError(
        "1-D Liu–Pierce AGHQ requires a scalar (dimension 1) integral; got dimension $dim. " *
        "Tensor / multi-d AGHQ is not implemented (#448)."))
    return nothing
end

"""
    _aghq_1d_logint(logf, mode, hess; k=5)
    _aghq_1d_logint(logf, mode, hess, z, w)

Log of `∫ exp(logf(x)) dx` by Liu–Pierce (1994) adaptive Gauss–Hermite
quadrature, wrapping `_gauss_hermite`.

`mode` maximises `logf`; `hess = logf''(mode)` must be negative. Nodes and
weights `(z, w)` are the physicists' rule
`∫ h(x) e^{-x²} dx ≈ Σ wₖ h(zₖ)`. The adaptive map is
`x = mode + √2 σ z` with `σ = 1/√(−hess)`, so

    ∫ exp(logf) ≈ √2 σ Σₖ wₖ exp(logf(xₖ) + zₖ²).

`k=1` (node 0, weight `√π`) is exactly the 1-point Laplace approximation
`logf(mode) + ½ log(2π) − ½ log(−hess)`. That is a plumbing identity, not a
quadrature or recovery claim.

A non-scalar `mode` fails loud (`dim ≠ 1`). Multi-d / tensor AGHQ is out of
scope for #448.
"""
function _aghq_1d_logint(logf, mode::Real, hess::Real; k::Int = 5)
    k >= 1 || throw(ArgumentError("AGHQ node count k must be ≥ 1; got $k (#448)"))
    z, w = _gauss_hermite(Int(k))
    return _aghq_1d_logint(logf, mode, hess, z, w)
end

function _aghq_1d_logint(logf, mode::Real, hess::Real,
                        z::AbstractVector, w::AbstractVector)
    _aghq_require_1d(1)
    T = promote_type(typeof(mode), typeof(hess), eltype(z), Float64)
    hessT = convert(T, hess)
    hessT < zero(T) || throw(ArgumentError(
        "AGHQ Hessian at the mode must be negative (log-concave); got $hess (#448)"))
    σ = one(T) / sqrt(-hessT)
    k = length(z)
    k == length(w) || throw(ArgumentError("AGHQ nodes and weights must match in length (#448)"))
    terms = Vector{T}(undef, k)
    rt2 = sqrt(convert(T, 2))
    @inbounds for i in 1:k
        zi = convert(T, z[i])
        xi = convert(T, mode) + rt2 * σ * zi
        terms[i] = logf(xi) + zi * zi + log(convert(T, w[i]))
    end
    mx = maximum(terms)
    return log(rt2) + log(σ) + mx + log(sum(exp(t - mx) for t in terms))
end

function _aghq_1d_logint(logf, mode::AbstractVector, hess; k::Int = 5)
    _aghq_require_1d(length(mode))
    h = hess isa AbstractMatrix ? hess[1] : hess
    return _aghq_1d_logint(logf, mode[1], h; k = k)
end

# Per-group log-integral returned when a Poisson group's posterior cannot be
# evaluated (overflowing η0, non-finite or non-negative curvature, non-finite
# quadrature). It is a large *finite* constant, so the outer optimiser's line
# search sees a very bad objective and backs off instead of `drm` aborting
# (#1063). `_fit_poisson_ranef_aghq` throws a clear error if the optimum itself
# is penalised.
const _AGHQ_PENALTY_LOGINT = -1e8

"""
    _poisson_group_mode(y, η0, idx, invσ2) -> Float64

Mode `b̂` of the strictly concave Poisson `(1 | g)` group posterior
`b ↦ Σᵢ yᵢ(η0ᵢ + b) − exp(η0ᵢ + b) − ½ b² invσ2`, computed on primal values.

The score `s(b) = Σy − b·invσ2 − Σᵢ exp(η0ᵢ + b)` is strictly decreasing, and
the root lies between `0` and the likelihood-only solution
`b_lik = log Σy − logsumexp(η0)` (where `Σᵢ exp(η0ᵢ + b_lik) = Σy`, so no
exponential can overflow there). Newton's method is *safeguarded* inside that
bracket (bisection whenever a step leaves it or is non-finite), so a wild
`η0` — an outer line-search probe with a huge β — cannot overshoot, which the
old fixed 12-step Newton unroll from `b = 0` did (#1063). Returns `NaN` if no
finite bracket exists.
"""
function _poisson_group_mode(y, η0, idx, invσ2::Float64)
    sy = 0.0
    m = -Inf
    @inbounds for i in idx
        sy += y[i]
        e = _aghq_primal(η0[i])
        m = max(m, e)
    end
    isfinite(m) && isfinite(invσ2) && invσ2 > 0 || return NaN
    function score(b)
        s = sy - b * invσ2
        @inbounds for i in idx
            s -= exp(_aghq_primal(η0[i]) + b)
        end
        return s
    end
    # b_lik by logsumexp (finite: m is finite)
    lse = 0.0
    @inbounds for i in idx
        lse += exp(_aghq_primal(η0[i]) - m)
    end
    lse = m + log(lse)
    lo, hi, b = 0.0, 0.0, 0.0
    if sy > 0
        blik = log(sy) - lse
        s0 = score(0.0)
        if s0 == 0
            return 0.0
        elseif s0 > 0
            lo, hi, b = 0.0, max(blik, 0.0), blik       # root in (0, b_lik)
        else
            lo, hi, b = min(blik, 0.0), 0.0, blik       # root in (b_lik, 0)
        end
    else                                                  # all-zero group: root < 0
        hi = 0.0; lo = -1.0
        for _ in 1:80
            score(lo) > 0 && break
            lo *= 2
        end
        score(lo) > 0 || return NaN
        b = lo
    end
    isfinite(b) || return NaN
    for _ in 1:200
        s = score(b)
        if s > 0
            lo = b
        elseif s < 0
            hi = b
        else
            return b
        end
        h = -invσ2
        @inbounds for i in idx
            h -= exp(_aghq_primal(η0[i]) + b)
        end
        bn = b - s / h
        if !(isfinite(bn) && lo < bn < hi)
            bn = (lo + hi) / 2
        end
        abs(bn - b) <= 1e-14 * (1 + abs(bn)) && return bn
        b = bn
        hi - lo <= 1e-15 * (1 + abs(b)) && return b
    end
    return b
end

"""
    _poisson_group_aghq_logint(y, η0, lf, idx, σb, k)

Log `∫ L(y_g | b) φ(b; 0, σb²) db` for one Poisson `(1 | g)` group via 1-D
Liu–Pierce AGHQ.

The group posterior is strictly concave. Its mode is found on primal values
by a bracketed, safeguarded Newton solve (`_poisson_group_mode`), then three
Newton steps in the working type `T` attach exact first and second
`ForwardDiff` derivatives (implicit-function theorem), so the outer nll stays
smooth for the gradient and for the `ForwardDiff.hessian` used for the vcov.
If the posterior cannot be evaluated (non-finite curvature or result) the
group returns the large finite `_AGHQ_PENALTY_LOGINT` instead of throwing, so
the outer optimiser rejects the step (#1063). `k=1` recovers 1-point Laplace
of this integrand — plumbing, not a recovery claim.
"""
function _poisson_group_aghq_logint(y, η0, lf, idx, σb, k::Integer)
    T = promote_type(eltype(η0), typeof(σb))
    penalty = convert(T, _AGHQ_PENALTY_LOGINT)
    invσ2 = one(T) / (σb * σb)
    b0 = _poisson_group_mode(y, η0, idx, _aghq_primal(invσ2))
    isfinite(b0) || return penalty
    b = convert(T, b0)
    @inbounds for _ in 1:3
        g = -b * invσ2
        h = -invσ2
        for i in idx
            e = exp(η0[i] + b)
            g += y[i] - e
            h -= e
        end
        b -= g / h
    end
    hess = -invσ2
    @inbounds for i in idx
        hess -= exp(η0[i] + b)
    end
    (isfinite(_aghq_primal(hess)) && _aghq_primal(hess) < 0 &&
     isfinite(_aghq_primal(b))) || return penalty
    function logf(x)
        ll = -convert(T, 0.5) * log(convert(T, 2π)) - log(σb) -
             convert(T, 0.5) * x * x * invσ2
        @inbounds for i in idx
            η = η0[i] + x
            ll += y[i] * η - exp(η) - lf[i]
        end
        return ll
    end
    r = _aghq_1d_logint(logf, b, hess; k = Int(k))
    return isfinite(_aghq_primal(r)) ? r : penalty
end
