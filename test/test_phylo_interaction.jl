# Bipartite two-tree interaction random effect (host × parasite) on the Gaussian
# mean. The interaction effect has covariance σ²·(C_A ⊗ C_B); with one obs per
# (host, parasite) cell the marginal is exactly Gaussian:
#     y ~ N(Xβ, V),   V = σ²·(C_A ⊗ C_B) + σ_e²·I.
# Tests: (1) parameter recovery from a known σ²(C_A⊗C_B)+σ_e²I, and (2) a
# finite-difference gradient gate (≤ 1e-6) on the marginal logLik vs ForwardDiff
# AND vs central differences.
using DRModels
using Test, Random, LinearAlgebra
using ForwardDiff: gradient as fd_gradient

@testset "phylo_interaction: bipartite two-tree RE" begin

    @testset "covariance recovery (n_A=n_B=8 → 64 obs)" begin
        Random.seed!(20260610)
        nA = 8; nB = 8; n = nA * nB

        # C_A, C_B from small random trees (built by sigma_phy_dense → unit-diag
        # correlation), one tree per side of the bipartite interaction.
        phyA = random_balanced_tree(nA; branch_length = 0.35)
        phyB = random_balanced_tree(nB; branch_length = 0.45)
        C_A = phylo_correlation(phyA)
        C_B = phylo_correlation(phyB)
        @test size(C_A) == (nA, nA)
        @test size(C_B) == (nB, nB)
        @test isposdef(Symmetric(C_A))
        @test isposdef(Symmetric(C_B))

        # True marginal: V = σ²·(C_A⊗C_B) + σ_e²·I. Simulate y directly from V's
        # Cholesky so recovery targets the exact data-generating covariance.
        β0 = 0.7
        σ = 0.9        # interaction SD  (σ²  = 0.81)
        σe = 0.4       # residual SD     (σ_e² = 0.16)
        CAB = kron(C_A, C_B)
        Vtrue = σ^2 .* CAB .+ σe^2 .* Matrix(I, n, n)
        L = cholesky(Symmetric(Vtrue)).L
        X = ones(n, 1)
        y = β0 .+ L * randn(n)

        fit = fit_phylo_interaction(y, X, C_A, C_B; group = :interaction)

        @test is_converged(fit)
        @test isfinite(loglik(fit))
        # Intercept (mean is independent of the residual-correlation structure).
        @test coef(fit, :mu)[1] ≈ β0 atol = 0.6
        # Variance components (single realisation → generous tolerances; the
        # estimator is unbiased in expectation, this checks it lands in the basin).
        @test re_sd(fit)[:interaction] ≈ σ atol = 0.4
        @test exp(coef(fit, :sigma)[1]) ≈ σe atol = 0.25
    end

    @testset "FD gradient gate (≤ 1e-6) on the marginal logLik" begin
        Random.seed!(424242)
        nA = 8; nB = 8; n = nA * nB
        phyA = random_balanced_tree(nA; branch_length = 0.3)
        phyB = random_balanced_tree(nB; branch_length = 0.5)
        C_A = phylo_correlation(phyA)
        C_B = phylo_correlation(phyB)
        CAB = kron(C_A, C_B)

        σ = 0.8; σe = 0.5; β0 = 0.3
        Vtrue = σ^2 .* CAB .+ σe^2 .* Matrix(I, n, n)
        L = cholesky(Symmetric(Vtrue)).L
        X = ones(n, 1)
        y = β0 .+ L * randn(n)

        pμ = size(X, 2)
        # Marginal NLL as a pure function of θ = [βμ; log σ_e; log σ]; the gate
        # checks the analytic-via-AD path against an INDEPENDENT central-difference
        # gradient, away from any boundary (so V is well inside the PD cone).
        f(θ) = phylo_interaction_nll(θ, y, X, CAB; pμ = pμ)
        θ = [0.25, log(0.45), log(0.85)]   # off the optimum, interior point

        g_ad = fd_gradient(f, θ)           # ForwardDiff (the "exact" reference)
        # Central differences as the independent oracle.
        g_cd = similar(θ)
        for k in eachindex(θ)
            h = 1e-6 * max(abs(θ[k]), 1.0)
            θp = copy(θ); θm = copy(θ)
            θp[k] += h; θm[k] -= h
            g_cd[k] = (f(θp) - f(θm)) / (2h)
        end
        err = maximum(abs.(g_ad .- g_cd))
        @info "phylo_interaction FD gate" max_abs_grad_error = err
        @test err ≤ 1e-6
    end
end
