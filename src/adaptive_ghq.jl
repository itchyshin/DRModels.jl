# adaptive_ghq.jl — per-group adaptive Gauss–Hermite quadrature (AGHQ) for a
# q-dimensional Gaussian random effect that enters a linear predictor (#834).
#
# Used by every non-Gaussian `_fit_*_corr_ranef` route, `(1 + x | g)`. For group g
#
#     ∫ exp{ f_g(b) } db,   f_g(b) = Σ_{i∈g} ℓ_i(η0_i + z_iᵀ b) + log N(b; 0, Σ),
#
# with Σ = L Lᵀ. The grid is centred on each group's conditional mode b̂_g and scaled
# by the curvature there (Liu & Pierce 1994; Pinheiro & Bates 1995): with
# H_g = −∇²f_g(b̂_g) and C Cᵀ = H_g⁻¹, the nodes are b = b̂_g + √2 C z and
#
#     log ∫ exp f_g ≈ (q/2) log 2 + log|C| + log Σ_z w_z exp{ f_g(b̂_g + √2 C z) + zᵀz }.
#
# K = 1 node per axis is exactly the Laplace approximation
# f_g(b̂) + (q/2) log 2π − ½ log|H_g| (the drmTMB/TMB integrator for these models).
#
# The only family-specific input is `ll(i, η)`: the log-density of observation i at
# linear predictor η (any Real type; captured distributional parameters may be duals).
#
# Differentiability (ForwardDiff through the outer optimiser and the vcov Hessian):
# the mode is found by a safeguarded Newton ascent, then TWO further exact Newton
# steps are taken in the caller's number type. At a fixed point the Newton map has
# zero Jacobian in b, so one exact step makes ∂b̂/∂θ equal the implicit-function
# derivative −H⁻¹ ∂g/∂θ and a second makes ∂²b̂/∂θ² exact as well. The result is an
# exact gradient and Hessian of the AGHQ objective (to the 1e-10 mode tolerance)
# without hand-coded implicit derivatives. Per-observation ℓ′ and ℓ″ come from
# nested `ForwardDiff.derivative` (tags keep them separate from the outer duals).
#
# Written from the published method; no drmTMB (GPL) source is used.

"""
    _AGHQRule(q, K)

Tensor-product Gauss–Hermite rule for `q` dimensions, `K` nodes per axis. `Z` is
`q × K^q`; `lw[j] = Σ log wₖ + zⱼᵀzⱼ` folds the `exp(zᵀz)` AGHQ factor into the
weight.
"""
struct _AGHQRule
    q::Int
    K::Int
    Z::Matrix{Float64}
    lw::Vector{Float64}
end

function _AGHQRule(q::Integer, K::Integer)
    q >= 1 || throw(ArgumentError("AGHQ dimension q must be ≥ 1; got $q"))
    K >= 1 || throw(ArgumentError("AGHQ node count K must be ≥ 1; got $K"))
    z, w = _gauss_hermite(Int(K))
    lw1 = log.(w)
    N = Int(K)^Int(q)
    Z = Matrix{Float64}(undef, q, N)
    lw = Vector{Float64}(undef, N)
    for (j, ci) in enumerate(CartesianIndices(ntuple(_ -> Int(K), Int(q))))
        s = 0.0
        for d in 1:q
            zd = z[ci[d]]
            Z[d, j] = zd
            s += lw1[ci[d]] + zd * zd
        end
        lw[j] = s
    end
    return _AGHQRule(Int(q), Int(K), Z, lw)
end

# Default nodes per axis for the (1 + x | g) routes (#834). Sweep on the Poisson and
# Beta-binomial DGPs of test/test_adaptive_ghq.jl (G = 150, 20 obs/group), logLik
# error at the fitted optimum vs AGHQ-40: K=3 −0.19/−0.12, K=4 −0.003/−0.008,
# K=5 −0.003/−0.003, K=7 <1e-4 nat. K = 4 is the first inside 0.01 nat but only just
# (Beta-binomial); K = 5 keeps a ~3× margin and includes the mode node (odd K), at
# 25 nodes per group — still ~6× fewer than the old non-adaptive 12×12 grid.
const _CORR_RANEF_AGHQ_K = 5

# Default nodes for the 1-D `(1 | g)` random-intercept routes (#719). Before this,
# every non-Gaussian `_fit_*_ranef` route integrated b_g on a fixed 32-node
# PRIOR-scale grid (b = √2 σ_b z), independent of where the group posterior actually
# sits — on an informative Poisson group (HSAUR3::epilepsy, sd(subject) ≈ 0.52) the
# reported logLik landed 1.59 nat off the true marginal at DRModels.jl's optimum and
# the fit itself 5.2 nat below the true maximum. Swept on the same q=1 helper against
# K=61 on the informative-group DGP of test/test_adaptive_ghq_1d.jl (G=100, 30
# obs/group, RE SD 0.8): K=3 misses by 0.04–0.20 nat depending on family; K=5 is
# within 0.01 nat everywhere (worst case Poisson at −0.0034 nat), matching
# `_CORR_RANEF_AGHQ_K` above. LogNormal's `(1 | g)` marginal is linear-Gaussian in
# b_g, so AGHQ is exact there for any K (K=1 included) — see `_fit_lognormal_ranef`.
const _RANEF1D_AGHQ_K = 5

_aghq_primal(x::ForwardDiff.Dual) = _aghq_primal(ForwardDiff.value(x))
_aghq_primal(x::Real) = Float64(x)

# Lower Cholesky factor of a small symmetric matrix; `ok = false` if not PD.
function _aghq_chol(H::AbstractMatrix{T}) where {T}
    q = size(H, 1)
    Lc = zeros(T, q, q)
    for j in 1:q
        s = H[j, j]
        for k in 1:(j-1)
            s -= Lc[j, k]^2
        end
        _aghq_primal(s) > 0 || return Lc, false
        Lc[j, j] = sqrt(s)
        for i in (j+1):q
            t = H[i, j]
            for k in 1:(j-1)
                t -= Lc[i, k] * Lc[j, k]
            end
            Lc[i, j] = t / Lc[j, j]
        end
    end
    return Lc, true
end

# Solve (Lc Lcᵀ) x = r for lower-triangular Lc.
function _aghq_cholsolve(Lc::AbstractMatrix, r::AbstractVector)
    q = length(r)
    T = promote_type(eltype(Lc), eltype(r))
    u = Vector{T}(undef, q)
    for i in 1:q
        s = r[i]
        for k in 1:(i-1)
            s -= Lc[i, k] * u[k]
        end
        u[i] = s / Lc[i, i]
    end
    x = Vector{T}(undef, q)
    for i in q:-1:1
        s = u[i]
        for k in (i+1):q
            s -= Lc[k, i] * x[k]
        end
        x[i] = s / Lc[i, i]
    end
    return x
end

# Group log-integrand f_g(b) (prior included, constant −(q/2)log 2π − log|L|).
function _aghq_f(ll, idx, η0, Zre, Linv, logdetL, b)
    q = length(b)
    s = zero(promote_type(eltype(b), eltype(Linv)))
    for i in idx
        e = η0[i]
        for d in 1:q
            e += Zre[i, d] * b[d]
        end
        s += ll(i, e)
    end
    # ½‖L⁻¹ b‖²
    qf = zero(eltype(s))
    for r in 1:q
        u = zero(eltype(s))
        for c in 1:r
            u += Linv[r, c] * b[c]
        end
        qf += u * u
    end
    return s - qf / 2 - q * log(2π) / 2 - logdetL
end

# Gradient and NEGATIVE Hessian of f_g at b. `clip = true` drops observation
# curvature of the wrong sign (non-log-concave families such as Student-t) so the
# ascent direction stays well defined; the exact Newton steps use `clip = false`.
function _aghq_grad_neghess(ll, idx, η0, Zre, Sinv, b, clip::Bool)
    q = length(b)
    T = promote_type(eltype(b), eltype(Sinv))
    g = zeros(T, q)
    H = Matrix{T}(undef, q, q)
    for r in 1:q, c in 1:q
        H[r, c] = Sinv[r, c]
    end
    for r in 1:q
        for c in 1:q
            g[r] -= Sinv[r, c] * b[c]
        end
    end
    for i in idx
        e = η0[i]
        for d in 1:q
            e += Zre[i, d] * b[d]
        end
        d1f = u -> ForwardDiff.derivative(v -> ll(i, v), u)
        d1 = d1f(e)
        d2 = ForwardDiff.derivative(d1f, e)
        w = clip ? (_aghq_primal(d2) < 0 ? -d2 : zero(d2)) : -d2
        for r in 1:q
            zr = Zre[i, r]
            g[r] += d1 * zr
            for c in 1:q
                H[r, c] += w * zr * Zre[i, c]
            end
        end
    end
    return g, H
end

"""
    _aghq_group_logint(ll, idx, η0, Zre, L, rule, bstart; maxiter = 100, tol = 1e-10)

Log of `∫ exp(f_g(b)) db` for one group (see the file header), by adaptive
Gauss–Hermite quadrature with `rule` (`_AGHQRule`). `L` is the lower Cholesky
factor of the random-effect covariance, `Zre[i, :]` the random-effect design row of
observation `i`, and `bstart` a Float64 warm start for the mode. Returns
`(logint, b̂_primal)`.
"""
function _aghq_group_logint(ll, idx, η0, Zre, L::AbstractMatrix, rule::_AGHQRule,
                            bstart::AbstractVector{<:Real}; maxiter::Int = 100, tol::Real = 1e-10)
    q = rule.q
    size(L) == (q, q) || throw(DimensionMismatch("L must be $(q)×$(q)"))
    i1 = first(idx)
    T = promote_type(eltype(η0), eltype(L), typeof(ll(i1, η0[i1])))
    # Σ⁻¹ = L⁻ᵀ L⁻¹ via the lower-triangular inverse of L.
    Linv = zeros(T, q, q)
    for c in 1:q
        Linv[c, c] = one(T) / L[c, c]
        for r in (c+1):q
            s = zero(T)
            for k in c:(r-1)
                s += L[r, k] * Linv[k, c]
            end
            Linv[r, c] = -s / L[r, r]
        end
    end
    Sinv = transpose(Linv) * Linv
    logdetL = sum(log(L[d, d]) for d in 1:q)
    f(b) = _aghq_f(ll, idx, η0, Zre, Linv, logdetL, b)

    # 1. Safeguarded Newton ascent to the mode (clipped curvature + step halving).
    # Warm start from the cached mode unless the prior mean b = 0 is already better
    # (a cache left by a line-search probe at an extreme θ can be far off).
    b = T.(bstart)
    fb = _aghq_primal(f(b))
    b0 = zeros(T, q); f0 = _aghq_primal(f(b0))
    if !(fb >= f0) && !isnan(f0)
        b = b0; fb = f0
    end
    for _ in 1:maxiter
        isfinite(fb) || break
        g, H = _aghq_grad_neghess(ll, idx, η0, Zre, Sinv, b, true)
        Hc, ok = _aghq_chol(H)
        ok || break                                   # prior alone keeps H PD; defensive
        δ = _aghq_cholsolve(Hc, g)
        all(isfinite, _aghq_primal.(δ)) || break
        step = one(T); bn = b .+ δ; fn = _aghq_primal(f(bn))
        k = 0
        while !(fn >= fb - 1e-12 * abs(fb)) && k < 40
            step /= 2; bn = b .+ step .* δ; fn = _aghq_primal(f(bn)); k += 1
        end
        fn >= fb - 1e-12 * abs(fb) || break           # no ascent possible: b is the mode to tolerance
        b = bn; fb = fn
        maximum(abs, _aghq_primal.(step .* δ)) < tol && break
    end
    # 2. Two exact Newton steps: exact first and second θ-derivatives of b̂.
    for _ in 1:2
        g, H = _aghq_grad_neghess(ll, idx, η0, Zre, Sinv, b, false)
        Hc, ok = _aghq_chol(H)
        ok || break
        bn = b .+ _aghq_cholsolve(Hc, g)
        all(isfinite, _aghq_primal.(bn)) || break
        b = bn
    end
    # 3. Curvature at the mode; fall back to clipped curvature if not PD.
    _, H = _aghq_grad_neghess(ll, idx, η0, Zre, Sinv, b, false)
    Hc, ok = _aghq_chol(H)
    if !ok
        _, H = _aghq_grad_neghess(ll, idx, η0, Zre, Sinv, b, true)
        Hc, ok = _aghq_chol(H)
    end
    # Non-finite curvature (e.g. a line-search probe at an extreme θ where ℓ″ is
    # NaN): report an impossible point instead of evaluating nodes on garbage.
    ok || return convert(T, -Inf), _aghq_primal.(b)
    # C = Hc⁻ᵀ satisfies C Cᵀ = H⁻¹; log|C| = −Σ log diag(Hc).
    logdetC = -sum(log(Hc[d, d]) for d in 1:q)
    N = size(rule.Z, 2)
    terms = Vector{T}(undef, N)
    rt2 = sqrt(2.0)
    bn = Vector{T}(undef, q)
    for j in 1:N
        # u = Hc⁻ᵀ z (back-substitution with the upper-triangular Hcᵀ)
        for r in q:-1:1
            s = rule.Z[r, j] + zero(T)
            for k in (r+1):q
                s -= Hc[k, r] * bn[k]
            end
            bn[r] = s / Hc[r, r]
        end
        for r in 1:q
            bn[r] = b[r] + rt2 * bn[r]
        end
        tj = f(bn) + rule.lw[j]
        terms[j] = isnan(_aghq_primal(tj)) ? convert(T, -Inf) : tj
    end
    mx = maximum(terms)
    isfinite(_aghq_primal(mx)) || return convert(T, -Inf), _aghq_primal.(b)
    lse = mx + log(sum(exp(t - mx) for t in terms))
    return q * log(2.0) / 2 + logdetC + lse, _aghq_primal.(b)
end

"""
    _aghq_marginal_loglik(ll, members, η0, Zre, L, rule, bcache)

Sum over groups of `_aghq_group_logint`. `bcache` (`q × G`, Float64) holds each
group's last mode and is updated in place as a warm start; it affects only the
iteration count, not the converged value (mode tolerance 1e-10).
"""
function _aghq_marginal_loglik(ll, members, η0, Zre, L, rule::_AGHQRule, bcache::AbstractMatrix{Float64})
    tot = nothing
    for (gi, idx) in enumerate(members)
        isempty(idx) && continue
        v, bhat = _aghq_group_logint(ll, idx, η0, Zre, L, rule, @view(bcache[:, gi]))
        if all(isfinite, bhat)
            bcache[:, gi] .= bhat
        else
            bcache[:, gi] .= 0.0
        end
        tot = tot === nothing ? v : tot + v
    end
    return tot === nothing ? zero(eltype(η0)) : tot
end

# Lower Cholesky factor of the (1 + x | g) covariance in the `vc` log-Cholesky
# convention L = [exp(a) 0; cc exp(b)].
_corr_ranef_L(a, b, cc) = (l11 = exp(a); l22 = exp(b); z = zero(promote_type(typeof(l11), typeof(cc)));
                           [l11 z; cc l22])
