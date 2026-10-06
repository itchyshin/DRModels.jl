# Laplace marginal for the non-Gaussian location–scale model (#202 groundwork).
# Verifies the marginal assembly against a 2-D Gauss–Hermite integral: with many
# observations per group the posterior is near-Gaussian and Laplace → exact, so a
# sign or constant error in (jn + ½logdetH − ½logdetP) shows up immediately.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# Direct 2-D Gauss–Hermite marginal NLL for a SINGLE group, prior a ~ N(0, Λ).
# Uses the same _gauss_hermite rule the engine ships (weights sum to √π):
#   ∫ f(a) N(a;0,Λ) da ≈ (1/π) Σ_jk w_j w_k f(√2 L [z_j,z_k]),  Λ = LLᵀ.
function _ghq_marginal_nll(kind, y, η0, ψ0, Λ; K = 24)
    z, w = DRModels._gauss_hermite(K)
    L = cholesky(Symmetric(Λ)).L
    rt2 = sqrt(2.0)
    terms = Float64[]
    for j in 1:K, k in 1:K
        a = rt2 .* (L * [z[j], z[k]])
        ll = 0.0
        for i in eachindex(y)
            ll += -DRModels._ls_nll(kind, y[i], η0[i] + a[1], ψ0[i] + a[2])
        end
        push!(terms, log(w[j]) + log(w[k]) + ll)
    end
    m = maximum(terms)
    loglik = m + log(sum(exp.(terms .- m))) - log(π)
    return -loglik
end

@testset "location–scale Laplace marginal vs 2-D Gauss–Hermite (single group)" begin
    Random.seed!(20260606)
    Λ = DRModels._ls_lc_to_Λ([log(0.5), 0.12, log(0.4)])
    P = DRModels.prior_precision(sparse(1.0 * I, 1, 1), inv(Λ))   # = Λ⁻¹ (one group)
    L = cholesky(Symmetric(Λ)).L

    # NB2: many obs per group so Laplace ≈ exact.
    let m = 60, gidx = ones(Int, 60)
        η0 = fill(0.3, m); ψ0 = fill(0.7, m)
        a = L * randn(2)
        y = [Float64(rand(Distributions.NegativeBinomial(
                 exp(ψ0[i] + a[2]),
                 exp(ψ0[i] + a[2]) / (exp(ψ0[i] + a[2]) + exp(η0[i] + a[1]))
             ))) for i in 1:m]
        nll_lap, _, ok = DRModels._ls_marginal_nll(Val(:nb2), y, η0, ψ0, gidx, 1, P)
        nll_ghq = _ghq_marginal_nll(Val(:nb2), y, η0, ψ0, Λ)
        @test ok
        @test nll_lap ≈ nll_ghq rtol = 1e-2
    end

    # Gamma: same check.
    let m = 60, gidx = ones(Int, 60)
        η0 = fill(0.2, m); ψ0 = fill(0.9, m)
        a = L * randn(2)
        y = [rand(Distributions.Gamma(exp(ψ0[i] + a[2]), exp(η0[i] + a[1]) / exp(ψ0[i] + a[2]))) for i in 1:m]
        nll_lap, _, ok = DRModels._ls_marginal_nll(Val(:gamma), y, η0, ψ0, gidx, 1, P)
        nll_ghq = _ghq_marginal_nll(Val(:gamma), y, η0, ψ0, Λ)
        @test ok
        @test nll_lap ≈ nll_ghq rtol = 1e-2
    end
end

@testset "location–scale marginal: multi-group i.i.d. factorises over groups" begin
    # With an i.i.d. prior P = kron(I_G, Λ⁻¹) the marginal is the sum of the
    # per-group GHQ marginals — a second independent cross-check.
    Random.seed!(20260607)
    Λ = DRModels._ls_lc_to_Λ([log(0.45), 0.1, log(0.5)])
    G = 5; m = 40; n = G * m
    gidx = repeat(1:G, inner = m)
    P = DRModels.prior_precision(sparse(1.0 * I, G, G), inv(Λ))
    L = cholesky(Symmetric(Λ)).L
    η0 = fill(0.25, n); ψ0 = fill(0.6, n)
    y = Vector{Float64}(undef, n)
    for g in 1:G
        a = L * randn(2)
        for i in ((g-1)*m+1):(g*m)
            r = exp(ψ0[i] + a[2]); μ = exp(η0[i] + a[1])
            y[i] = Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ))))
        end
    end
    nll_lap, _, ok = DRModels._ls_marginal_nll(Val(:nb2), y, η0, ψ0, gidx, G, P)
    @test ok

    nll_ghq_total = 0.0
    for g in 1:G
        idx = ((g-1)*m+1):(g*m)
        nll_ghq_total += _ghq_marginal_nll(Val(:nb2), y[idx], η0[idx], ψ0[idx], Λ)
    end
    @test nll_lap ≈ nll_ghq_total rtol = 1e-2
end
