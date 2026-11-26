# cov lane 1125n11 — make_problem_from_Q validation + AugProblem masks (src/sparse_em_fit.jl).
using DRModels
using Test, LinearAlgebra, Random, SparseArrays

@testset "cov 1125n11: make_problem_from_Q errors" begin
    G = 4
    Q = Matrix{Float64}(I, G, G)
    n = G
    y1 = randn(n)
    y2 = randn(n)
    X1 = ones(n, 1)
    X2 = ones(n, 1)
    Xs1 = reshape(ones(n), n, 1)
    Xs2 = reshape(ones(n), n, 1)
    Xr = zeros(n, 0)
    group = 1:n
    @test_throws ErrorException DRModels.make_problem_from_Q(Q, y1[1:(end - 1)], y2, X1, X2, Xs1, Xs2, Xr; group = group)
    @test_throws ErrorException DRModels.make_problem_from_Q(Q, y1, y2, X1, X2, Xs1, Xs2, Xr; group = [1, 2])
    @test_throws ErrorException DRModels.make_problem_from_Q(Q, y1, y2, X1[1:(end - 1), :], X2, Xs1, Xs2, Xr; group = group)
    @test_throws ErrorException DRModels.make_problem_from_Q(Q, y1, y2, X1, X2, Xs1, Xs2, Xr; group = [0, 1, 2, 5])
    Qbad = [1.0 2.0; 2.0 1.0]
    @test_throws ErrorException DRModels.make_problem_from_Q(Qbad, y1[1:2], y2[1:2], X1[1:2, :], X2[1:2, :],
        Xs1[1:2, :], Xs2[1:2, :], Xr[1:2, :]; group = [1, 2])
end

@testset "cov 1125n11: make_problem_from_Q happy path and NaN masks" begin
    Random.seed!(1125119)
    G = 5
    Qd = Matrix{Float64}(I, G, G)
    Qd[1, 2] = Qd[2, 1] = -0.4
    n = G
    group = 1:G
    y1 = randn(n)
    y2 = randn(n)
    y1[3] = NaN
    X1 = hcat(ones(n), randn(n))
    X2 = ones(n, 1)
    Xs1 = reshape(ones(n), n, 1)
    Xs2 = reshape(ones(n), n, 1)
    Xr = zeros(n, 0)
    prob, Q = DRModels.make_problem_from_Q(Qd, y1, y2, X1, X2, Xs1, Xs2, Xr; group = group)
    @test prob isa AugProblem
    @test prob.p == G
    @test prob.n_total == G
    @test length(prob.leaf_node) == n
    @test !prob.obs1[3]
    @test prob.obs2[3]
    @test issparse(Q)
    @test size(Q, 1) == G
    β = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
         s1 = [-0.1], s2 = [-0.2], rho = Float64[])
    η1, η2, ηs1, ηs2, ηr = DRModels.leaf_etas(prob, β)
    @test length(η1) == n
    @test all(isfinite, η1)
    @test all(isfinite, η2)
    u = zeros(4)
    ll_full = DRModels.leaf_nll(u, prob.y1[1], prob.y2[1], η1[1], η2[1], ηs1[1], ηs2[1], 0.0, true, true)
    ll_y1 = DRModels.leaf_nll(u, prob.y1[1], prob.y2[1], η1[1], η2[1], ηs1[1], ηs2[1], 0.0, true, false)
    ll_none = DRModels.leaf_nll(u, prob.y1[1], prob.y2[1], η1[1], η2[1], ηs1[1], ηs2[1], 0.0, false, false)
    @test isfinite(ll_full)
    @test isfinite(ll_y1)
    @test ll_none == 0.0
    P = prior_precision(Q, inv(Matrix(Symmetric(0.3I(4)))))
    @test issparse(P)
    @test nnz(P) > 0
end

@testset "cov 1125n11: pack_theta / unpack_theta roundtrip" begin
    Random.seed!(1125120)
    p = 5
    phy = random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q = phy.Q_topology[keep, keep]
    pos = Dict(n => i for (i, n) in enumerate(keep))
    leaf_node = [pos[phy.leaf_indices[k]] for k in 1:p]
    n = p
    prob = AugProblem(phy, length(keep), p, leaf_node, randn(n), randn(n),
                      ones(n, 1), ones(n, 1), reshape(ones(n), n, 1), reshape(ones(n), n, 1), reshape(ones(n), n, 1))
    Λ = Matrix(Symmetric(0.25 * I(4)))
    β = (mu1 = [0.1], mu2 = [-0.2], s1 = [-0.5], s2 = [-0.4], rho = [0.2])
    θ = pack_theta(β, Λ)
    @test length(θ) == DRModels.theta_len(prob)
    β2, lc2 = unpack_theta(prob, θ)
    @test β2.mu1 ≈ β.mu1
    @test only(β2.rho) ≈ only(β.rho)
    @test lc_to_Λ(lc2) ≈ Λ rtol = 1e-12
end