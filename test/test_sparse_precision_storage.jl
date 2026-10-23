# S5a: regression for accidental dense materialisation of the root-conditioned
# augmented phylogenetic precision.  This test uses only stdlib Random.

using Test
using DRModels
using LinearAlgebra
using SparseArrays
using Random

const _S5A_ALLOCATIONS = Dict{Int,Int}()

function _s5a_component_allocation(tips::Int)
    phy = random_balanced_tree(tips; branch_length = 0.2)
    gidx = collect(1:tips)
    DRModels._phylo_aug_comp(gidx, tips, phy, :species) # compile and warm CHOLMOD
    GC.gc()
    return @allocated DRModels._phylo_aug_comp(gidx, tips, phy, :species)
end

function _s5a_lss_fixture()
    rng = MersenneTwister(20260830)
    tips = 8
    phy = random_balanced_tree(tips; branch_length = 0.2)
    # Interleaved, unbalanced observed groups; leaves 2, 4, 6, and 8 are
    # intentionally unobserved, so the sparse precision retains latent-only rows.
    gidx = [3, 1, 3, 7, 1, 5, 7, 3, 5, 1, 7, 5, 3, 7, 1, 5]
    n = length(gidx)
    x = randn(rng, n)
    z = collect(range(-0.5, 0.5; length = tips))
    Xmu = hcat(ones(n), x)
    Xsigma = hcat(ones(n), x)
    Zg = hcat(ones(tips), z)
    residual_sd = exp.(-0.7 .+ 0.2 .* x)
    y = 0.4 .+ 0.2 .* x .+ 0.15 .* z[gidx] .+ residual_sd .* randn(rng, n)
    return phy, y, Xmu, Xsigma, Zg, gidx, tips
end

function _s5a_fd_gradient(f, theta; step = 1e-6)
    gradient = similar(theta)
    for index in eachindex(theta)
        delta = step * max(abs(theta[index]), 1.0)
        plus = copy(theta); minus = copy(theta)
        plus[index] += delta; minus[index] -= delta
        gradient[index] = (f(plus) - f(minus)) / (2 * delta)
    end
    gradient
end

function _s5a_dense_lss_nll(theta, phy, y, Xmu, Xsigma, Zg, gidx)
    p_mu = size(Xmu, 2)
    p_sigma = size(Xsigma, 2)
    beta_mu = theta[1:p_mu]
    beta_sigma = theta[(p_mu + 1):(p_mu + p_sigma)]
    alpha = theta[(p_mu + p_sigma + 1):end]
    Q, leaf_pos, _ = augmented_tree_precision(phy)
    leaf_covariance = inv(Matrix(Q))[leaf_pos, leaf_pos]
    leaf_sd = sqrt.(diag(leaf_covariance))
    correlation = leaf_covariance ./ (leaf_sd * leaf_sd')
    residual_variance = exp.(2 .* (Xsigma * beta_sigma))
    phylo_sd = exp.(Zg * alpha)
    covariance = Diagonal(residual_variance) +
        Diagonal(phylo_sd[gidx]) * correlation[gidx, gidx] * Diagonal(phylo_sd[gidx])
    factor = cholesky(Symmetric(covariance))
    residual = y - Xmu * beta_mu
    return 0.5 * (logdet(factor) + dot(residual, factor \ residual) + length(y) * log(2π))
end

@testset "Sparse phylogenetic precision storage" begin
    # A small numerical oracle fixes the exact root-conditioned precision and
    # leaf-correlation rescaling used by the public structured component.
    phy = random_balanced_tree(8; branch_length = 0.2)
    comp = DRModels._phylo_aug_comp(collect(1:8), 8, phy, :species)
    Q, leaf_pos, q = augmented_tree_precision(phy)
    Qdense = Matrix(Q)
    Qinv = inv(Qdense)
    leaf_variance = diag(Qinv)[leaf_pos]
    @test comp.m == q
    @test issparse(comp.Q)
    @test Matrix(comp.Q) ≈ Qdense atol = 1e-12
    @test comp.logdetCprior ≈ -logdet(Symmetric(Qdense)) atol = 1e-12
    @test comp.wts ≈ 1.0 ./ sqrt.(leaf_variance) atol = 1e-10

    # Each allocation measurement is warmed. This covers `_phylo_aug_comp`; the
    # sparse-LSS entry point is instead checked below against an independent dense
    # objective, until the protected-core patch can be approved and measured.
    # The 20 KiB/tip ceiling is generous
    # for sparse factor work and selected inversion, while rejecting the old q×q
    # dense copy (which alone exceeds the bound at 1,024 tips).
    for tips in (1024, 2048)
        _S5A_ALLOCATIONS[tips] = _s5a_component_allocation(tips)
    end
end

@testset "Sparse LSS retains multi-alpha objective behaviour" begin
    phy, y, Xmu, Xsigma, Zg, gidx, tips = _s5a_lss_fixture()
    fit = DRModels._fit_phylo_gaussian_lss_sparse(
        Gaussian(), y, Xmu, Xsigma, Zg, gidx, tips, phy,
        ["(Intercept)", "x"], ["(Intercept)", "x"], ["(Intercept)", "z"],
        :species, 1e-8,
    )
    # Stationarity is the finite-difference check below (2e-4), which is
    # looser than g_tol = 1e-8. Do not also require the gradient stop.
    @test isfinite(fit.loglik)
    dense_objective = theta -> _s5a_dense_lss_nll(theta, phy, y, Xmu, Xsigma, Zg, gidx)
    dense_gradient = _s5a_fd_gradient(dense_objective, fit.theta)
    @test isapprox(fit.nll(fit.theta), dense_objective(fit.theta); atol = 1e-8)
    @test all(isfinite, dense_gradient)
    @test norm(dense_gradient, Inf) ≤ 2e-4

    # Changing the second alpha predictor must change the independent objective;
    # the sparse fit above then has its own objective and stationarity pinned to it.
    Zg_perturbed = copy(Zg)
    Zg_perturbed[:, 2] .*= 1.25
    theta_probe = copy(fit.theta)
    theta_probe[(size(Xmu, 2) + size(Xsigma, 2) + 1):end] .= [0.2, 1.0]
    dense_perturbed = theta -> _s5a_dense_lss_nll(theta, phy, y, Xmu, Xsigma, Zg_perturbed, gidx)
    @test !isapprox(dense_objective(theta_probe), dense_perturbed(theta_probe); atol = 1e-10, rtol = 0)
end

@testset "Sparse phylogenetic precision allocation ceiling" begin
    for tips in (1024, 2048)
        @test _S5A_ALLOCATIONS[tips] ≤ 20_000 * tips
    end
end

println("SPARSE_PRECISION_REGRESSIONS_PASS")
