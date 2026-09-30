# locscale_marginal.jl — Laplace marginal for the non-Gaussian location–scale
# model (#202). Groundwork only: not wired into `drm()`.
#
# Given fixed-effect parts η0 = Xβ and ψ0 = Zγ, a grouping, and the prior
# precision P = kron(Q, Λ⁻¹), the marginal integrates out the q=2 latent a:
#
#   p(y) = ∫ [∏ᵢ p(yᵢ | a)] N(a; 0, P⁻¹) da.
#
# The Laplace approximation expands the joint jn(a) = −Σ log p(yᵢ|a) + ½ aᵀPa at
# its mode â (the inner solve). With H = ∇²jn(â) the marginal NLL is
#
#   −log p(y) ≈ jn(â) + ½ logdet H − ½ logdet P,
#
# (the 2π factors of the prior and the Laplace integral cancel). This mirrors the
# verified q=4 PLSM marginal, reduced to q=2 with a non-Gaussian data term.
# `test/test_locscale_marginal.jl` checks it against a 2-D Gauss–Hermite integral
# (Laplace → exact as obs-per-group grows).

using LinearAlgebra: logdet, cholesky, Symmetric, issuccess

"""
    _ls_marginal_nll(kind, y, η0, ψ0, gidx, G, P; a0=nothing, tol=1e-9)

Laplace-approximate marginal negative log-likelihood for the q=2 location–scale
model. Returns `(nll, â, ok)`: the marginal NLL, the inner mode, and a success
flag (the inner Newton solve can fail at extreme parameters). `tol` is the inner
mode's stationarity bound (`_ls_inner_mode`).
"""
function _ls_marginal_nll(kind, y, η0, ψ0, gidx, G, P,
                          Zη = _ls_canonical_Zeta(length(y)),
                          Zψ = _ls_canonical_Zpsi(length(y)); a0 = nothing,
                          tol::Real = 1e-9)
    a, ch, ok = _ls_inner_mode(kind, y, η0, ψ0, gidx, G, P, Zη, Zψ; a0 = a0, tol = tol)
    ok || return Inf, a, false
    jn = _ls_joint(kind, y, η0, ψ0, gidx, a, P, Zη, Zψ)
    logdetH = logdet(ch)
    # The prior precision P = kron(Q, Λ⁻¹) is PD in exact arithmetic, but profiling
    # a variance/covariance parameter toward its boundary drives Λ near-singular,
    # so Λ⁻¹ can overflow and the factorisation fail numerically. Treat that as an
    # infeasible point (the documented `ok = false` path) instead of throwing.
    chP = cholesky(Symmetric(P); check = false)
    issuccess(chP) || return Inf, a, false
    logdetP = logdet(chP)
    return jn + 0.5 * logdetH - 0.5 * logdetP, a, true
end
