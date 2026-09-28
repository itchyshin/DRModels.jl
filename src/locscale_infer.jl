# locscale_infer.jl — Wald inference + group-level summaries for the q=2
# location–scale fit (#202). Builds on the exact outer gradient
# (`_ls_marginal_grad`): the observed information is the symmetric
# finite-difference Jacobian of that gradient at θ̂. Because the gradient is exact
# and O(p), and the number of packed parameters is fixed, the whole Hessian costs
# O(p) — no dense p×p Hessian of the latent field is ever formed.

using LinearAlgebra: Symmetric, diag, SingularException

"""
    _ls_obs_information(kind, y, Xμ, Xψ, gidx, G, Q, θ; h=1e-5) -> Symmetric

Observed information ∂²M/∂θ² of the Laplace marginal at θ, as the symmetrised
central finite-difference Jacobian of the exact gradient `_ls_marginal_grad`.
"""
function _ls_obs_information(kind, y, Xμ, Xψ, gidx, G, Q, θ,
                            Zη = _ls_canonical_Zeta(length(y)),
                            Zψ = _ls_canonical_Zpsi(length(y)); h = 1e-5, a0 = nothing)
    p = length(θ)
    H = zeros(p, p)
    @inbounds for k in 1:p
        θp = copy(θ); θp[k] += h
        θm = copy(θ); θm[k] -= h
        gp = _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θp, Zη, Zψ; a0 = a0)
        gm = _ls_marginal_grad(kind, y, Xμ, Xψ, gidx, G, Q, θm, Zη, Zψ; a0 = a0)
        H[:, k] = (gp .- gm) ./ (2h)
    end
    return Symmetric((H + H') ./ 2)
end

# Wald covariance V = (observed information)⁻¹. Returns `nothing` if the
# information is singular (e.g. a variance pinned at the boundary) or
# non-finite (e.g. the finite-difference gradient blows up as Λ approaches a
# singular mean-scale correlation, #870 review). `inv` on a matrix with
# NaN/Inf entries throws `ArgumentError`, not `SingularException`, so that
# case is guarded explicitly before `inv` is ever called -- matching
# `_ls_whitened_vcov`'s convention (locscale_whitened.jl) of returning
# `nothing` at the same near-singular-Λ boundary. Downstream callers already
# normalise a `nothing` vcov to NaN SEs (`locscale_frontend.jl`,
# `locscale_corr.jl`).
function _ls_vcov(kind, y, Xμ, Xψ, gidx, G, Q, θ,
                  Zη = _ls_canonical_Zeta(length(y)),
                  Zψ = _ls_canonical_Zpsi(length(y)); h = 1e-5, a0 = nothing)
    H = _ls_obs_information(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ; h = h, a0 = a0)
    Hm = Matrix(H)
    if !all(isfinite, Hm)
        @warn "location-scale Wald vcov: observed information is not finite " *
              "(likely a near-singular Λ / boundary mean-scale correlation) -- " *
              "returning no vcov/SEs for this fit."
        return nothing
    end
    return try
        inv(Hm)
    catch err
        if err isa SingularException
            @warn "location-scale Wald vcov: observed information is singular " *
                  "at the optimum (boundary Λ) -- returning no vcov/SEs for this fit."
            nothing
        else
            rethrow(err)
        end
    end
end

# Standard errors from a covariance matrix; NaN where the variance is non-positive.
_ls_se(V) = V === nothing ? nothing : [d > 0 ? sqrt(d) : NaN for d in diag(V)]

"""
    _ls_components(Λ) -> NamedTuple

Named group-level summaries of the 2×2 covariance Λ: the mean-axis SD, the
scale-axis SD, and the mean↔scale correlation ρ_a.
"""
function _ls_components(Λ)
    sμ = sqrt(Λ[1, 1]); sψ = sqrt(Λ[2, 2])
    return (sd_mu = sμ, sd_psi = sψ, cor_mu_psi = Λ[1, 2] / (sμ * sψ))
end
