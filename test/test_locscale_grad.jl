# Exact O(p) outer gradient for the q=2 location–scale Laplace marginal (#202).
# The decisive gate: the analytic `_ls_marginal_grad` must match central finite
# differences of the SAME packed objective `_ls_fit_nll` — at a generic θ (NOT
# the optimum, so every component is exercised) — for NB2 and Gamma families,
# under both an i.i.d. prior and a phylogenetic-tree prior. A sign/index error in
# any of the βμ / βψ / λ blocks (or the third-derivative or selected-inverse
# terms) shows up here. Derivation: docs/dev-log/2026-06-06-locscale-exact-gradient.md.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# Central finite-difference gradient of a scalar θ-function.
function _fd_grad(f, θ; h = 1e-6)
    g = similar(θ)
    for k in eachindex(θ)
        θp = copy(θ); θp[k] += h
        θm = copy(θ); θm[k] -= h
        g[k] = (f(θp) - f(θm)) / (2h)
    end
    return g
end

# Relative agreement check robust to the differing magnitudes across blocks.
_grad_ok(ga, gfd; tol = 1e-4) = maximum(abs.(ga .- gfd)) < tol * (1 + maximum(abs.(gfd)))

@testset "location–scale exact gradient vs finite differences" begin

    @testset "NB2, i.i.d. groups" begin
        Random.seed!(101)
        G = 5; m = 6; n = G * m
        gidx = repeat(1:G, inner = m)
        x = randn(n); z = randn(n)
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)
        Λt = DRModels._ls_lc_to_Λ([log(0.4), 0.1, log(0.5)])
        Lt = cholesky(Symmetric(Λt)).L
        A = [Lt * randn(2) for _ in 1:G]
        y = [(r = exp(0.2 + A[gidx[i]][2]); μ = exp(0.3 + 0.4x[i] + A[gidx[i]][1]);
              Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ))))) for i in 1:n]
        Q = sparse(1.0 * I, G, G)
        θ = [0.25, 0.35, 0.1, -0.05, log(0.45), 0.08, log(0.55)]   # generic, not optimum

        f = θ -> DRModels._ls_fit_nll(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θ)
        ga = DRModels._ls_marginal_grad(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θ)
        gfd = _fd_grad(f, θ)
        @test all(isfinite, ga) && all(ga .!= 0)  # finite, populated gradient
        @test all(isfinite, gfd)
        @test _grad_ok(ga, gfd)
    end

    @testset "Gamma, i.i.d. groups" begin
        Random.seed!(202)
        G = 5; m = 6; n = G * m
        gidx = repeat(1:G, inner = m)
        x = randn(n); z = randn(n)
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)
        Λt = DRModels._ls_lc_to_Λ([log(0.35), 0.05, log(0.4)])
        Lt = cholesky(Symmetric(Λt)).L
        A = [Lt * randn(2) for _ in 1:G]
        y = [(α = exp(0.5 + A[gidx[i]][2]); μ = exp(0.2 + 0.3x[i] + A[gidx[i]][1]);
              rand(Distributions.Gamma(α, μ / α))) for i in 1:n]
        Q = sparse(1.0 * I, G, G)
        θ = [0.15, 0.25, 0.4, 0.06, log(0.4), 0.05, log(0.45)]

        f = θ -> DRModels._ls_fit_nll(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, θ)
        ga = DRModels._ls_marginal_grad(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, θ)
        gfd = _fd_grad(f, θ)
        @test all(isfinite, ga) && all(ga .!= 0)
        @test all(isfinite, gfd)
        @test _grad_ok(ga, gfd)
    end

    @testset "NB2, phylogenetic-tree prior" begin
        Random.seed!(303)
        p = 6; m = 6; n = p * m
        phy = random_balanced_tree(p; branch_length = 0.25)
        C = sigma_phy_dense(phy; σ²_phy = 1.0)
        LC = cholesky(Symmetric(C)).L
        Λt = DRModels._ls_lc_to_Λ([log(0.4), 0.0, log(0.3)])
        LΛ = cholesky(Symmetric(Λt)).L
        A = LC * randn(p, 2) * LΛ'
        species = repeat(1:p, inner = m)
        x = randn(n); z = randn(n)
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)
        y = [(r = exp(0.2 + A[species[i], 2]); μ = exp(0.15 + 0.4x[i] + A[species[i], 1]);
              Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ))))) for i in 1:n]
        Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
        θ = [0.2, 0.3, 0.1, 0.04, log(0.42), 0.05, log(0.32)]

        f = θ -> DRModels._ls_fit_nll(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θ)
        ga = DRModels._ls_marginal_grad(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θ)
        gfd = _fd_grad(f, θ)
        @test all(isfinite, ga) && all(ga .!= 0)
        @test all(isfinite, gfd)
        @test _grad_ok(ga, gfd)
    end
end
