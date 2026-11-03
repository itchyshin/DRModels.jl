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

# Reference correlation scaling, independent of the code under test.
cov2cor_ref(Σ) = Σ ./ sqrt.(diag(Σ) * diag(Σ)')

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
    # Shared helper: simulate y from V = σ²(C_A⊗C_B) + σ_e² I on a nA×nB grid.
    function _sim_interaction(seed, nA, nB; σ = 0.9, σe = 0.4, β0 = 0.5)
        Random.seed!(seed)
        C_A = phylo_correlation(random_balanced_tree(nA; branch_length = 0.35))
        C_B = phylo_correlation(random_balanced_tree(nB; branch_length = 0.45))
        n = nA * nB
        CAB = kron(C_A, C_B)
        L = cholesky(Symmetric(σ^2 .* CAB .+ σe^2 .* Matrix(I, n, n))).L
        X = ones(n, 1)
        y = β0 .+ L * randn(n)
        return (; C_A, C_B, CAB, X, y, n)
    end

    @testset "phylo_correlation: structure and hand-computed values" begin
        for phy in (random_balanced_tree(8; branch_length = 0.3),
                    random_balanced_tree(5; branch_length = 0.2),
                    random_caterpillar_tree(7; branch_length = 0.25))
            C = phylo_correlation(phy)
            p = phy.n_leaves
            @test size(C) == (p, p)
            @test C ≈ C'
            @test all(diag(C) .≈ 1.0)
            @test all(0 .- 1e-12 .≤ C .≤ 1 .+ 1e-12)
            Σ = sigma_phy_dense(phy; σ²_phy = 1.0)
            @test C ≈ cov2cor_ref(Σ)
        end

        # Newick input equals the AugmentedPhy route.
        nwk = "((A:0.1,B:0.2):0.3,C:0.5);"
        @test phylo_correlation(nwk) ≈ phylo_correlation(augmented_phy(nwk))

        # Hand computation: Var(A)=0.1+0.3, Var(B)=0.2+0.3, Var(C)=0.5,
        # Cov(A,B)=0.3 (shared branch), all other covariances 0.
        C3 = phylo_correlation(nwk)
        @test C3[1, 2] ≈ 0.3 / sqrt(0.4 * 0.5)
        @test C3[1, 3] ≈ 0.0 atol = 1e-12
        @test C3[2, 3] ≈ 0.0 atol = 1e-12

        # Two tips joined only at the root share no path: correlation 0.
        C2 = phylo_correlation("(A:0.1,B:0.2);")
        @test C2 ≈ Matrix(I, 2, 2) atol = 1e-12
    end

    @testset "fit_phylo_interaction: input validation errors" begin
        d = _sim_interaction(11, 4, 3)
        # length(y) not equal to n_A·n_B
        @test_throws ErrorException fit_phylo_interaction(
            d.y[1:end-1], d.X[1:end-1, :], d.C_A, d.C_B)
        @test_throws r"length\(y\)" fit_phylo_interaction(
            d.y[1:end-1], d.X, d.C_A, d.C_B)
        # grid implied by the C matrices disagrees with y (swapped sizes ok,
        # different product is not)
        C_A2 = phylo_correlation(random_balanced_tree(5; branch_length = 0.3))
        @test_throws r"length\(y\)" fit_phylo_interaction(d.y, d.X, C_A2, d.C_B)
        # X with the wrong row count
        @test_throws ErrorException fit_phylo_interaction(
            d.y, d.X[1:end-1, :], d.C_A, d.C_B)
        @test_throws r"X has" fit_phylo_interaction(
            d.y, ones(d.n + 2, 1), d.C_A, d.C_B)
    end

    @testset "group is only the variance-component label" begin
        # The docstring documents `group` as the key under which the interaction
        # SD is reported by re_sd; it must not change the likelihood or CAB.
        d = _sim_interaction(21, 6, 5)
        f1 = fit_phylo_interaction(d.y, d.X, d.C_A, d.C_B)                 # :interaction
        f2 = fit_phylo_interaction(d.y, d.X, d.C_A, d.C_B; group = :hostpara)
        @test haskey(re_sd(f1), :interaction)
        @test !haskey(re_sd(f1), :hostpara)
        @test haskey(re_sd(f2), :hostpara)
        @test !haskey(re_sd(f2), :interaction)
        @test loglik(f1) ≈ loglik(f2) atol = 1e-8
        @test re_sd(f1)[:interaction] ≈ re_sd(f2)[:hostpara] rtol = 1e-6
        # The structure itself is C_A ⊗ C_B (not C_B ⊗ C_A): swapping the roles
        # of the two matrices gives a different likelihood when n_A != n_B.
        fswap = fit_phylo_interaction(d.y, d.X, d.C_B, d.C_A)
        @test abs(loglik(fswap) - loglik(f1)) > 1e-6
    end

    @testset "nll: finite at optimum, minimal there, row-permutation invariant" begin
        d = _sim_interaction(31, 6, 9)
        fit = fit_phylo_interaction(d.y, d.X, d.C_A, d.C_B)
        @test is_converged(fit)
        θ̂ = copy(fit.theta)
        f(θ) = phylo_interaction_nll(θ, d.y, d.X, d.CAB; pμ = 1)
        f0 = f(θ̂)
        @test isfinite(f0)
        # Kronecker order: row (a-1)·n_B + b  <=>  kron(C_A, C_B), not kron(C_B, C_A).
        @test -loglik(fit) ≈ f0 atol = 1e-8
        @test norm(kron(d.C_A, d.C_B) - kron(d.C_B, d.C_A)) > 1e-3
        for k in (2, 3), δ in (-0.5, 0.5)       # log σ_e and log σ
            θp = copy(θ̂); θp[k] += δ
            @test f(θp) > f0
        end
        # Consistent permutation of y, X rows and CAB rows/cols leaves nll fixed.
        perm = randperm(d.n)
        fperm = phylo_interaction_nll(θ̂, d.y[perm], d.X[perm, :],
                                      d.CAB[perm, perm]; pμ = 1)
        @test fperm ≈ f0 rtol = 1e-10
    end

    @testset "convergence on another seed and with n_A != n_B" begin
        for (seed, nA, nB) in ((777, 8, 8), (20261019, 6, 9), (99, 9, 6))
            d = _sim_interaction(seed, nA, nB)
            fit = fit_phylo_interaction(d.y, d.X, d.C_A, d.C_B)
            @test is_converged(fit)
            @test isfinite(loglik(fit))
            @test all(isfinite, fit.theta)
            @test re_sd(fit)[:interaction] > 0
            @test 0.05 < exp(coef(fit, :sigma)[1]) < 1.5
            @test re_sd(fit)[:interaction] ≈ 0.9 atol = 0.6
        end
    end
end
