# cov lane 1125n10 — engine-core whitened prior path (src/sparse_aug_plsm.jl).
using DRModels
using Test, LinearAlgebra, Random, SparseArrays

function _cov_n10_q4_fixture(; p::Int = 5, seed::Int = 112530)
    Random.seed!(seed)
    phy = DRModels.random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    pos = Dict(n => i for (i, n) in enumerate(keep))
    leaf_node = [pos[phy.leaf_indices[k]] for k in 1:p]
    n = p
    X1 = ones(n, 1)
    X2 = ones(n, 1)
    Xs1 = reshape(ones(n), n, 1)
    Xs2 = reshape(ones(n), n, 1)
    Xr = zeros(n, 0)
    y1 = randn(n)
    y2 = randn(n)
    prob = DRModels.AugProblem(phy, length(keep), p, leaf_node, y1, y2, X1, X2, Xs1, Xs2, Xr)
    return prob, Q_cond
end

@testset "cov 1125n10: whitened coordinate roundtrip" begin
    prob, Q = _cov_n10_q4_fixture()
    Λ = Matrix(Symmetric([0.4 0.08 0.0 0.0; 0.08 0.35 0.0 0.0; 0.0 0.0 0.15 0.02; 0.0 0.0 0.02 0.14]))
    L = cholesky(Λ).L
    W = DRModels.whitened_prior(Q, L)
    @test W isa DRModels.WhitenedPrior
    @test issparse(W.Pw)
    u = randn(4 * prob.n_total)
    v = DRModels.u_to_whitened(W, u)
    u2 = DRModels.whitened_to_u(W, v)
    @test u2 ≈ u rtol = 1e-12
    v2 = DRModels.u_to_whitened(W, u2)
    @test v2 ≈ v rtol = 1e-12
end

@testset "cov 1125n10: joint_nll sparse vs whitened" begin
    prob, Q = _cov_n10_q4_fixture(seed = 112531)
    Λ = Matrix(Symmetric(0.3I(4)))
    P = DRModels.prior_precision(Q, inv(Λ))
    W = DRModels.whitened_prior(Q, cholesky(Λ).L)
    β = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
         s1 = [-0.2], s2 = [-0.25], rho = Float64[])
    u = randn(4 * prob.n_total)
    v = DRModels.u_to_whitened(W, u)
    nll_u = DRModels.joint_nll(prob, P, u, β)
    nll_v = DRModels.joint_nll(prob, W, v, β)
    @test isfinite(nll_u)
    @test isfinite(nll_v)
    @test nll_u ≈ nll_v rtol = 1e-9 atol = 1e-8
    g_u = DRModels.joint_grad(prob, P, u, β)
    g_v = DRModels.joint_grad(prob, W, v, β)
    @test all(isfinite, g_u)
    @test all(isfinite, g_v)
    u_from_g = DRModels.whitened_to_u(W, v)
    @test norm(g_u - DRModels.joint_grad(prob, P, u_from_g, β)) < 1e-7 * max(1.0, norm(g_u))
end

@testset "cov 1125n10: build_Huu and aug_prior_grad!" begin
    prob, Q = _cov_n10_q4_fixture(seed = 112532)
    Λ = Matrix(Symmetric(0.32I(4)))
    P = DRModels.prior_precision(Q, inv(Λ))
    nnz_full = nnz(P)
    @test nnz_full >= 4 * 4 * nnz(Q)
    W = DRModels.whitened_prior(Q, cholesky(Λ).L)
    β = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
         s1 = [-0.1], s2 = [-0.15], rho = Float64[])
    u, ch, _ = DRModels.estep_mode(prob, P, β; n_newton = 18)
    Hs = DRModels.build_Huu(prob, P, u, β)
    @test issymmetric(Matrix(Hs))
    v = DRModels.u_to_whitened(W, u)
    Hw = DRModels.build_Huu(prob, W, v, β)
    @test issymmetric(Matrix(Hw))
    g = zeros(length(u))
    DRModels.aug_prior_grad!(g, P, u)
    @test g ≈ P * u rtol = 1e-14
    gv = zeros(length(v))
    DRModels.aug_prior_grad!(gv, W.Pw, v)
    @test gv ≈ W.Pw * v rtol = 1e-14
    @test isposdef(Symmetric(Matrix(Hs) + 1e-6I))
end