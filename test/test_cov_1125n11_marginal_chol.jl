# cov lane 1125n11 — marginal_nll chol_ref reuse + build_Huu_expected (src/fit_q4_sparse_tmb.jl, sparse_aug_plsm.jl).
using DRModels
using Test, LinearAlgebra, Random, SparseArrays

function _n11_q4_small(; p::Int = 5, seed::Int = 1125121)
    Random.seed!(seed)
    phy = random_balanced_tree(p; branch_length = 0.18)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q = phy.Q_topology[keep, keep]
    pos = Dict(n => i for (i, n) in enumerate(keep))
    leaf_node = [pos[phy.leaf_indices[k]] for k in 1:p]
    n = p
    x = randn(n)
    prob = AugProblem(phy, length(keep), p, leaf_node, randn(n), randn(n),
                      hcat(ones(n), x), hcat(ones(n), 0.2 .* x),
                      reshape(ones(n), n, 1), reshape(ones(n), n, 1), reshape(ones(n), n, 1))
    return prob, Q
end

@testset "cov 1125n11: marginal_nll chol_ref stable on repeat" begin
    prob, Q = _n11_q4_small()
    β = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
         s1 = [-0.3], s2 = [-0.35], rho = [0.0])
    Λ = Matrix(Symmetric(0.32I(4)))
    θ = pack_theta(β, Λ)
    cache = DRModels.CholPatternCache()
    nll1, u1, ch1, W1 = marginal_nll(prob, Q, θ; n_newton = 18, chol_ref = cache)
    nll2, u2, ch2, W2 = marginal_nll(prob, Q, θ; u0 = u1, n_newton = 18, chol_ref = cache)
    @test isfinite(nll1)
    @test isfinite(nll2)
    @test nll1 ≈ nll2 rtol = 1e-10
    @test u1 ≈ u2 rtol = 1e-9
    @test W1 isa DRModels.WhitenedPrior
    @test W2 isa DRModels.WhitenedPrior
    @test length(u1) == 4 * prob.n_total
    @test ch1 !== nothing
    @test ch2 !== nothing
end

@testset "cov 1125n11: build_Huu vs build_Huu_expected at E-step mode" begin
    prob, Q = _n11_q4_small(seed = 1125122)
    Λ = Matrix(Symmetric([0.35 0.06 0.0 0.0; 0.06 0.32 0.0 0.0; 0.0 0.0 0.15 0.02; 0.0 0.0 0.02 0.14]))
    P = prior_precision(Q, inv(Λ))
    β = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
         s1 = [-0.2], s2 = [-0.25], rho = [0.0])
    u, ch, _ = estep_mode(prob, P, β; n_newton = 20)
    Hu = build_Huu(prob, P, u, β)
    He = DRModels.build_Huu_expected(prob, P, u, β)
    @test size(Hu) == size(He)
    @test issymmetric(Matrix(Hu))
    @test issymmetric(Matrix(He))
    @test norm(Hu - He) / (norm(Hu) + 1e-8) < 0.35
    ll = DRModels.laplace_ll(prob, P, β, u, ch)
    @test isfinite(ll)
    @test ll < 0
end

@testset "cov 1125n11: marginal_and_exact_grad length and finiteness" begin
    prob, Q = _n11_q4_small(seed = 1125123)
    β = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
         s1 = [-0.4], s2 = [-0.45], rho = [0.0])
    θ = pack_theta(β, Matrix(Symmetric(0.28I(4))))
    nll, g, u, ch = marginal_and_exact_grad(prob, Q, θ; n_newton = 22)
    @test isfinite(nll)
    @test length(g) == length(θ)
    @test all(isfinite, g)
    @test length(u) == 4 * prob.n_total
    H = lc_metric(prob, Q, θ, u; n_newton = 12)
    @test diag(H) |> minimum > 0
    g_lc = g[(length(g) - 9):length(g)]
    @test norm(H * (H \ g_lc) - g_lc) / (norm(g_lc) + 1e-8) < 1e-10
end