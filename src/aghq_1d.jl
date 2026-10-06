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

"""
    _poisson_group_aghq_logint(y, η0, lf, idx, σb, k)

Log `∫ L(y_g | b) φ(b; 0, σb²) db` for one Poisson `(1 | g)` group via 1-D
Liu–Pierce AGHQ.

The group posterior is strictly concave. The inner mode is a **fixed**
12-step Newton unroll (same discipline as the VA inner solve) so the outer
nll stays ForwardDiff-smooth. `k=1` recovers 1-point Laplace of this
integrand — plumbing, not a recovery claim.
"""
function _poisson_group_aghq_logint(y, η0, lf, idx, σb, k::Integer)
    T = promote_type(eltype(η0), typeof(σb))
    b = zero(T)
    invσ2 = one(T) / (σb * σb)
    @inbounds for _ in 1:12
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
    function logf(x)
        ll = -convert(T, 0.5) * log(convert(T, 2π)) - log(σb) -
             convert(T, 0.5) * x * x * invσ2
        @inbounds for i in idx
            η = η0[i] + x
            ll += y[i] * η - exp(η) - lf[i]
        end
        return ll
    end
    return _aghq_1d_logint(logf, b, hess; k = Int(k))
end
