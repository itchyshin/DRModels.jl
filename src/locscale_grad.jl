# locscale_grad.jl — exact O(p) outer gradient of the q=2 location–scale Laplace
# marginal (#202). Differentiates `_ls_fit_nll` w.r.t. the packed
# θ = [βμ; βψ; λ(3)] analytically, in O(p) (Takahashi selected inverse + a single
# adjoint solve; never forms a dense Hessian inverse).
#
# Derivation and the per-block formulas are in
# docs/dev-log/2026-06-06-locscale-exact-gradient.md. In brief, with
#   M(θ) = jn(â) + ½ logdet H − ½ logdet P,  g(â)=0,  v_j=½ tr(H⁻¹ ∂H/∂a_j),
#   w = H⁻¹ v,
# the exact gradient is
#   dM/dθₖ = ∂jn/∂θₖ + ½ tr(H⁻¹ ∂H/∂θₖ) − ½ tr(P⁻¹ ∂P/∂θₖ) − wᵀ ∂g/∂θₖ.
# Third derivatives of the kernel (needed for ∂H/∂a and ∂H/∂β) come from
# ForwardDiff of the analytic `_ls_hess`, so there is no hand-coded 3rd-deriv
# algebra. `test/test_locscale_grad.jl` gates this against central finite
# differences of `_ls_fit_nll` for i.i.d. and tree fixtures.

using ForwardDiff: derivative
using SparseArrays: rowvals, nonzeros, nzrange

# Per-observation third derivatives of the kernel in (η, ψ), obtained by
# ForwardDiff of the analytic second-derivative kernel. Returns
# (tηηη, tηηψ, tηψψ, tψψψ); the two mixed paths agree by symmetry.
function _ls_third(kind, y, η, ψ)
    dη = derivative(t -> collect(_ls_hess(kind, y, t, ψ)), η)   # d/dη (hηη,hηψ,hψψ)
    dψ = derivative(t -> collect(_ls_hess(kind, y, η, t)), ψ)   # d/dψ (hηη,hηψ,hψψ)
    return dη[1], dη[2], dη[3], dψ[3]
end

# ∂Λ/∂λ_k (2×2) by ForwardDiff of the log-Cholesky map.
function _dΛ_dλ(λ, k::Int)
    return derivative(t -> _ls_lc_to_Λ([i == k ? t : λ[i] for i in 1:3]), λ[k])
end

# Derivatives of L^{-T}L^{-1} in log-Cholesky coordinates. Construct them
# directly: -Λinv*dΛ*Λinv subtracts large nearly equal products when L22 is
# small, losing enough precision to reverse a profile nuisance gradient.
function _ls_precision_derivatives(λ)
    u, c, v = λ
    e1 = exp(-2 * u)
    e2 = exp(-2 * v)
    inv_diag_product = exp(-(u + v))
    scaled_c = c * inv_diag_product
    cross = exp(-u - 2 * v)
    A = e1 + scaled_c * scaled_c
    B = -c * cross
    z = zero(A)
    return ([-2 * A -B; -B z],
            [2 * scaled_c * inv_diag_product -cross; -cross z],
            [-2 * scaled_c * scaled_c -2 * B; -2 * B -2 * e2])
end

"""
    _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ) -> Vector

Exact gradient of `_ls_fit_nll` at the packed θ = [βμ; βψ; λ(3)]. Returns an
all-`NaN` vector if the inner Laplace mode fails to converge (rare; this is the
infeasibility signal that pairs with the `_ls_fit_nll` value sentinel — a
gradient-based optimiser rejects a NaN step rather than mistaking a zero gradient
for stationarity, see #314). O(p) in the number of groups. `a0` and `tol` are the
inner mode's start and stationarity bound (`_ls_inner_mode`).
"""
function _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ,
                           Zη = _ls_canonical_Zeta(length(y)),
                           Zψ = _ls_canonical_Zpsi(length(y)); a0 = nothing,
                           tol::Real = 1e-9, relaxed::Bool = false)
    pμ = size(Xμ, 2); pψ = size(Xψ, 2)
    βμ = @view θ[1:pμ]
    βψ = @view θ[pμ+1:pμ+pψ]
    λ  = θ[pμ+pψ+1:pμ+pψ+3]
    Λinv = _ls_lc_inv2x2(λ)   # stable: never forms Λ (see locscale_inner.jl)
    P = prior_precision(Q, Λinv)
    η0 = Xμ * βμ; ψ0 = Xψ * βψ

    a, ch, ok = _ls_inner_mode(kind, y, η0, ψ0, gidx, G, P, Zη, Zψ; a0 = a0, tol = tol,
                               relaxed = relaxed)
    # Inner-mode failure ⇒ infeasible θ. Return NaN (not zeros) so a gradient-based
    # optimiser treats the step as rejected instead of reading a zero gradient as
    # convergence at an infeasible point (#314). The paired value sentinel in
    # `_ls_fit_nll`/`_ls_profile_nll` covers the objective side.
    ok || return fill(NaN, length(θ))
    grad = zeros(length(θ))

    # 2×2 diagonal blocks of H⁻¹ from the Takahashi selected inverse.
    Hinv = takahashi_selinv(ch)
    α = zeros(G); β = zeros(G); δ = zeros(G)
    @inbounds for g in 1:G
        α[g] = Hinv[2g-1, 2g-1]; β[g] = Hinv[2g-1, 2g]; δ[g] = Hinv[2g, 2g]
    end

    # Adjoint v_j = ½ tr(H⁻¹ ∂H/∂a_j). With general loadings the per-obs block is
    # B_i = hηη·zη zηᵀ + hηψ·(zη zψᵀ+zψ zηᵀ) + hψψ·zψ zψᵀ, so its derivative
    # along latent slot j (direction d_j = (zη[c_j], zψ[c_j])) contracts the
    # kernel 3rd-deriv tensor with the per-group selected-inverse block M_g=[α β;
    # β δ] via the three scalars qηη=zηᵀM zη, qηψ=zηᵀM zψ, qψψ=zψᵀM zψ:
    #   v_j += ½ (dhηη·qηη + 2 dhηψ·qηψ + dhψψ·qψψ),  dhαβ = tαβη·dη + tαβψ·dψ.
    # The canonical loadings (zη=[1,0],zψ=[0,1]) recover the original v expression.
    v = zeros(2G)
    @inbounds for i in eachindex(y)
        g = gidx[i]
        a1 = a[2g-1]; a2 = a[2g]
        z1 = Zη[i, 1]; z2 = Zη[i, 2]; w1 = Zψ[i, 1]; w2 = Zψ[i, 2]
        ηi = η0[i] + z1 * a1 + z2 * a2
        ψi = ψ0[i] + w1 * a1 + w2 * a2
        t1, t2, t3, t4 = _ls_third(kind, y[i], ηi, ψi)   # tηηη,tηηψ,tηψψ,tψψψ
        αg = α[g]; βg = β[g]; δg = δ[g]
        qηη = αg * z1 * z1 + 2βg * z1 * z2 + δg * z2 * z2   # zηᵀ M zη
        qηψ = αg * z1 * w1 + βg * (z1 * w2 + z2 * w1) + δg * z2 * w2   # zηᵀ M zψ
        qψψ = αg * w1 * w1 + 2βg * w1 * w2 + δg * w2 * w2   # zψᵀ M zψ
        # slot 2g-1: direction d = (z1, w1); slot 2g: direction d = (z2, w2).
        for (slot, dη, dψ) in ((2g - 1, z1, w1), (2g, z2, w2))
            dhηη = t1 * dη + t2 * dψ
            dhηψ = t2 * dη + t3 * dψ
            dhψψ = t3 * dη + t4 * dψ
            v[slot] += 0.5 * (dhηη * qηη + 2 * dhηψ * qηψ + dhψψ * qψψ)
        end
    end
    w = ch \ v

    # βμ / βψ components (one obs loop). βμ shifts η0 (direction (1,0)); βψ shifts
    # ψ0 (direction (0,1)) — independent of the loadings. The adjoint enters via
    # wη = w_g·zη, wψ = w_g·zψ (the group adjoint contracted with each loading).
    @inbounds for i in eachindex(y)
        g = gidx[i]
        z1 = Zη[i, 1]; z2 = Zη[i, 2]; w1 = Zψ[i, 1]; w2 = Zψ[i, 2]
        ηi = η0[i] + z1 * a[2g-1] + z2 * a[2g]
        ψi = ψ0[i] + w1 * a[2g-1] + w2 * a[2g]
        gη, gψ = _ls_grad(kind, y[i], ηi, ψi)
        hηη, hηψ, hψψ = _ls_hess(kind, y[i], ηi, ψi)
        t1, t2, t3, t4 = _ls_third(kind, y[i], ηi, ψi)
        αg = α[g]; βg = β[g]; δg = δ[g]
        qηη = αg * z1 * z1 + 2βg * z1 * z2 + δg * z2 * z2
        qηψ = αg * z1 * w1 + βg * (z1 * w2 + z2 * w1) + δg * z2 * w2
        qψψ = αg * w1 * w1 + 2βg * w1 * w2 + δg * w2 * w2
        wη = w[2g-1] * z1 + w[2g] * z2     # w_g · zη
        wψ = w[2g-1] * w1 + w[2g] * w2     # w_g · zψ
        # mean-axis chain (direction (1,0) on the predictor pair):
        cμ = gη + 0.5 * (t1 * qηη + 2 * t2 * qηψ + t3 * qψψ) -
             (hηη * wη + hηψ * wψ)
        # scale-axis chain (direction (0,1) on the predictor pair):
        cψ = gψ + 0.5 * (t2 * qηη + 2 * t3 * qηψ + t4 * qψψ) -
             (hηψ * wη + hψψ * wψ)
        for k in 1:pμ
            grad[k] += Xμ[i, k] * cμ
        end
        for k in 1:pψ
            grad[pμ+k] += Xψ[i, k] * cψ
        end
    end

    # λ components: only P depends on λ, so ∂H/∂λ = ∂P/∂λ = kron(Q, Mk).
    qrows = rowvals(Q); qvals = nonzeros(Q)
    precision_derivatives = _ls_precision_derivatives(λ)
    @inbounds for k in 1:3
        Mk = precision_derivatives[k]         # ∂Λ⁻¹/∂λ_k
        t_quad = 0.0; t_adj = 0.0; t_tr = 0.0
        for h in 1:G
            for idx in nzrange(Q, h)
                g = qrows[idx]; q = qvals[idx]
                ag1 = a[2g-1]; ag2 = a[2g]; ah1 = a[2h-1]; ah2 = a[2h]
                m_ah1 = Mk[1, 1] * ah1 + Mk[1, 2] * ah2
                m_ah2 = Mk[2, 1] * ah1 + Mk[2, 2] * ah2
                t_quad += q * (ag1 * m_ah1 + ag2 * m_ah2)
                t_adj  += q * (w[2g-1] * m_ah1 + w[2g] * m_ah2)
                hb11 = Hinv[2g-1, 2h-1]; hb12 = Hinv[2g-1, 2h]
                hb21 = Hinv[2g, 2h-1];   hb22 = Hinv[2g, 2h]
                # ⟨Hinv_block(g,h), Mk⟩ = Σ_{s,t} Hinv[s,t]·Mk[t,s]
                t_tr += q * (hb11 * Mk[1, 1] + hb12 * Mk[2, 1] +
                             hb21 * Mk[1, 2] + hb22 * Mk[2, 2])
            end
        end
        # −½ logdet P = G*(logL11 + logL22) + a Q-only constant.
        # Its derivative is exact without a cancellation-prone trace product.
        normalization = k == 2 ? 0.0 : Float64(G)
        grad[pμ+pψ+k] = 0.5 * t_quad + 0.5 * t_tr - t_adj + normalization
    end

    return grad
end
