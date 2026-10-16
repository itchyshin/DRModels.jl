# j3_dense_reference.jl -- an INDEPENDENT second-derivative reference for the Laplace marginal
# of a binomial crossed-intercepts GLMM, used by test_j3a_gradient_hessian_vcov.jl (#1073).
#
# It shares no code with the route under test: dense matrices, an explicit Newton solve for the
# conditional mode in Float64, then three Newton steps carried in ForwardDiff Dual arithmetic
# (implicit-function theorem; each step adds one order of derivative accuracy, so the nested
# Duals of `ForwardDiff.hessian` see an exact b-hat(theta) to second order), and the Laplace
# value  F(b-hat, theta) + 1/2 log det H(b-hat, theta).  The Hessian of THAT function is the
# reference; nothing here differences a gradient or a value.
using ForwardDiff, LinearAlgebra

_pv(x::Real) = x
_pv(x::ForwardDiff.Dual) = _pv(ForwardDiff.value(x))

# negative joint log-density in (b; theta), up to constants that do not depend on theta
function _dr_joint_parts(b, θ, s, ntr, X, gidx, hidx, G, Hh)
    p = size(X, 2)
    T = promote_type(eltype(b), eltype(θ))
    β = θ[1:p]; invg = exp(-2 * θ[p+1]); invh = exp(-2 * θ[p+2])
    η0 = X * β
    q = G + Hh
    grad = zeros(T, q); Hmat = zeros(T, q, q); F = zero(T)
    for i in eachindex(s)
        gi = gidx[i]; hi = G + hidx[i]
        η = η0[i] + b[gi] + b[hi]
        pr = inv(one(T) + exp(-η))
        F += -(s[i] * log(pr) + (ntr[i] - s[i]) * log1p(-pr))
        r = ntr[i] * pr - s[i]; w = ntr[i] * pr * (1 - pr)
        grad[gi] += r; grad[hi] += r
        Hmat[gi, gi] += w; Hmat[hi, hi] += w; Hmat[gi, hi] += w; Hmat[hi, gi] += w
    end
    for j in 1:G;  F += 0.5 * invg * b[j]^2;  grad[j] += invg * b[j];  Hmat[j, j] += invg; end
    for j in (G+1):q; F += 0.5 * invh * b[j]^2; grad[j] += invh * b[j]; Hmat[j, j] += invh; end
    return F, grad, Hmat
end

function dense_binomial_crossed_nll(θ::AbstractVector, s, ntr, X, gidx, hidx, G, Hh)
    θ0 = _pv.(θ)
    b = zeros(G + Hh)
    for _ in 1:100                                   # Float64 mode, plain Newton
        _, g, Hm = _dr_joint_parts(b, θ0, s, ntr, X, gidx, hidx, G, Hh)
        δ = Hm \ g
        b -= δ
        norm(δ) < 1e-13 * (1 + norm(b)) && break
    end
    bT = convert(Vector{eltype(θ)}, b)
    for _ in 1:3                                     # carry derivatives through the mode
        _, g, Hm = _dr_joint_parts(bT, θ, s, ntr, X, gidx, hidx, G, Hh)
        bT = bT - Hm \ g
    end
    F, _, Hm = _dr_joint_parts(bT, θ, s, ntr, X, gidx, hidx, G, Hh)
    p = size(X, 2)
    return F + 0.5 * logdet(Hm) + G * θ[p+1] + Hh * θ[p+2]
end

dense_binomial_crossed_hessian(θ, args...) =
    ForwardDiff.hessian(t -> dense_binomial_crossed_nll(t, args...), θ)
