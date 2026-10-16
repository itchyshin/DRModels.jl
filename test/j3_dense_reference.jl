# j3_dense_reference.jl -- an INDEPENDENT second-derivative reference for Laplace-marginal
# GLMMs, used by the #1073 (lane J3) tests next to the native-drmTMB references.
#
# It shares no code with the routes under test: dense matrices, an explicit Newton solve for the
# conditional mode in Float64, then three Newton steps carried in ForwardDiff Dual arithmetic
# (implicit-function theorem; each step adds one order of derivative accuracy, so the nested
# Duals of `ForwardDiff.hessian` see an exact b-hat(theta) to second order), and the Laplace
# value  F(b-hat, theta) + 1/2 log det (Z' diag(w) Z + P) - 1/2 log det P.  The Hessian of THAT
# function is the reference; it differences neither a gradient nor a value of a route.
#
# A model is a `DenseSpec`:
#   p                    number of leading fixed-effect coefficients in theta
#   X, Z                 fixed-effect design (n x p) and random-effect loading (n x q)
#   obs_nll(eta, i, th)  -log p(y_i | eta) up to constants that do not depend on theta
#   prior(th)            (P, logdetP): prior PRECISION of b and its log determinant
using ForwardDiff, LinearAlgebra
using SpecialFunctions: loggamma

struct DenseSpec{F,G}
    p::Int
    X::Matrix{Float64}
    Z::Matrix{Float64}
    obs_nll::F
    prior::G
end

_pv(x::Real) = x
_pv(x::ForwardDiff.Dual) = _pv(ForwardDiff.value(x))

# per-observation first and second derivative of obs_nll in eta (by ForwardDiff, not by hand)
function _dr_rw(spec, η, θ)
    n = length(η)
    T = promote_type(eltype(η), eltype(θ))
    r = Vector{T}(undef, n); w = Vector{T}(undef, n)
    for i in 1:n
        f = e -> spec.obs_nll(e, i, θ)
        r[i] = ForwardDiff.derivative(f, η[i])
        w[i] = ForwardDiff.derivative(e -> ForwardDiff.derivative(f, e), η[i])
    end
    return r, w
end

function _dr_parts(spec, b, θ)
    η = spec.X * θ[1:spec.p] .+ spec.Z * b
    P, logdetP = spec.prior(θ)
    r, w = _dr_rw(spec, η, θ)
    F = sum(spec.obs_nll(η[i], i, θ) for i in eachindex(η)) + 0.5 * dot(b, P * b)
    grad = spec.Z' * r .+ P * b
    Hmat = spec.Z' * (w .* spec.Z) .+ P
    return F, grad, Hmat, logdetP
end

function dense_laplace_nll(θ::AbstractVector, spec::DenseSpec)
    θ0 = _pv.(θ)
    q = size(spec.Z, 2)
    b = zeros(q)
    for _ in 1:100                                   # Float64 mode, plain Newton
        _, g, Hm, _ = _dr_parts(spec, b, θ0)
        δ = Hm \ g
        b -= δ
        norm(δ) < 1e-13 * (1 + norm(b)) && break
    end
    bT = convert(Vector{eltype(θ)}, b)
    for _ in 1:3                                     # carry derivatives through the mode
        _, g, Hm, _ = _dr_parts(spec, bT, θ)
        bT = bT - Hm \ g
    end
    F, _, Hm, logdetP = _dr_parts(spec, bT, θ)
    return F + 0.5 * logdet(Hm) - 0.5 * logdetP
end

dense_laplace_hessian(θ, spec::DenseSpec) = ForwardDiff.hessian(t -> dense_laplace_nll(t, spec), θ)

# standard error from the reference Hessian
dense_laplace_se(θ, spec::DenseSpec) =
    (H = dense_laplace_hessian(θ, spec); sqrt.(diag(inv(Symmetric((H .+ H') ./ 2)))))

# ---- the model pieces the J3 tests need ---------------------------------------------------
_logistic(x) = inv(one(x) + exp(-x))

obs_poisson(y) = (η, i, θ) -> exp(η) - y[i] * η
obs_binomial(s, ntr) = (η, i, θ) -> (p = _logistic(η); -(s[i] * log(p) + (ntr[i] - s[i]) * log1p(-p)))
# NB2 with size r = exp(-2 psi), psi = theta[psi_index] (drmTMB: sigma^2 = 1/size)
function obs_nb2(y, psi_index)
    return function (η, i, θ)
        r = exp(-2 * θ[psi_index]); μ = exp(η)
        return -(loggamma(y[i] + r) - loggamma(r) + r * log(r) + y[i] * η - (r + y[i]) * log(r + μ))
    end
end

# blocks of independent iid effects: sds are exp(theta[sd_index[k]]), `sizes[k]` levels each
function prior_iid(sizes, sd_index)
    q = sum(sizes)
    return function (θ)
        d = Vector{eltype(θ)}(undef, q); ld = zero(eltype(θ)); o = 0
        for (k, m) in enumerate(sizes)
            invv = exp(-2 * θ[sd_index[k]])
            d[o+1:o+m] .= invv; ld += m * (-2 * θ[sd_index[k]]); o += m
        end
        return Diagonal(d), ld
    end
end

# one correlated block: b ~ N(0, exp(2 theta[sd_index]) C), C a known covariance matrix
function prior_cov(C, sd_index)
    Cinv = inv(Symmetric(C)); ldC = logdet(Symmetric(C))
    return function (θ)
        s2 = exp(2 * θ[sd_index])
        return Cinv ./ s2, -(size(C, 1) * 2 * θ[sd_index] + ldC)
    end
end

indicator(idx, G) = (Z = zeros(length(idx), G); for (i, g) in enumerate(idx); Z[i, g] = 1.0; end; Z)

# NB2 with a log-linear dispersion: psi_i = theta[i0] + theta[i0+1] * z_i, size r_i = exp(-2 psi_i)
function obs_nb2_hetero(y, z, i0)
    return function (η, i, θ)
        r = exp(-2 * (θ[i0] + θ[i0+1] * z[i])); μ = exp(η)
        return -(loggamma(y[i] + r) - loggamma(r) + r * log(r) + y[i] * η - (r + y[i]) * log(r + μ))
    end
end

# cumulative logit, K = 3 ordered categories y in 1:3; theta[i0] = first cutpoint, theta[i0+1] the
# log of the gap to the second (the ordered parameterisation drmTMB and DRModels both use)
function obs_cumlogit3(y, i0)
    return function (η, i, θ)
        c1 = θ[i0]; c2 = θ[i0] + exp(θ[i0+1])
        F(c) = _logistic(c - η)
        pr = y[i] == 1 ? F(c1) : (y[i] == 2 ? F(c2) - F(c1) : 1 - F(c2))
        return -log(pr)
    end
end

# Gaussian phylo mean, closed form (no Laplace needed): y ~ N(X beta, s2 I + p2 C),
# theta = [beta; log s; log s_phy]. ML or REML (the REML adds 1/2 logdet X'V^-1 X).
function dense_gaussian_phylo_nll(θ::AbstractVector, y, X, Cobs; reml::Bool = false)
    p = size(X, 2)
    V = exp(2 * θ[p+1]) * I + exp(2 * θ[p+2]) * Cobs
    F = cholesky(Symmetric(V))
    if reml
        VX = F \ X; A = X' * VX
        β = A \ (X' * (F \ y))
        e = y - X * β
        return 0.5 * (logdet(F) + dot(e, F \ e)) + 0.5 * logdet(Symmetric(A))
    end
    e = y - X * θ[1:p]
    return 0.5 * (logdet(F) + dot(e, F \ e))
end

# Gamma, log link, shape alpha = exp(-2 psi), psi = theta[psi_index] (drmTMB: sigma = CV)
function obs_gamma(y, psi_index)
    return function (η, i, θ)
        α = exp(-2 * θ[psi_index])
        return -(α * log(α) - α * η - loggamma(α) + (α - 1) * log(y[i]) - α * y[i] * exp(-η))
    end
end

# REML full-theta objective theta = [beta; log s; log s_phy]: the ML objective plus 1/2 logdet X'V^-1 X
function dense_gaussian_phylo_reml_full(θ::AbstractVector, y, X, Cobs)
    p = size(X, 2)
    V = exp(2 * θ[p+1]) * I + exp(2 * θ[p+2]) * Cobs
    F = cholesky(Symmetric(V))
    e = y - X * θ[1:p]
    A = X' * (F \ X)
    return 0.5 * (logdet(F) + dot(e, F \ e)) + 0.5 * logdet(Symmetric(A))
end
