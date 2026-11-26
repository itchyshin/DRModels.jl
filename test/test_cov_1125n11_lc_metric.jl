# cov lane 1125n11 — extra lc_metric contracts (src/lc_metric.jl); complements test_lc_metric.jl.
using DRModels
using Test, LinearAlgebra, Random, SparseArrays

function _n11_q4_fixture(; p::Int = 6, seed::Int = 1125111)
    Random.seed!(seed)
    phy = random_balanced_tree(p; branch_length = 0.22)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    pos = Dict(n => i for (i, n) in enumerate(keep))
    leaf_node = [pos[phy.leaf_indices[k]] for k in 1:p]
    n = p
    x = randn(n)
    X1 = hcat(ones(n), x)
    X2 = hcat(ones(n), -0.2 .* x)
    Xs1 = reshape(ones(n), n, 1)
    Xs2 = reshape(ones(n), n, 1)
    Xr = reshape(0.1 .* x, n, 1)
    y1 = 0.3 .+ x .+ 0.4 .* randn(n)
    y2 = -0.1 .+ 0.5 .* x .+ 0.5 .* randn(n)
    prob = AugProblem(phy, length(keep), p, leaf_node, y1, y2, X1, X2, Xs1, Xs2, Xr)
    return prob, Q_cond
end

@testset "cov 1125n11: lc_metric structure and kwargs" begin
    prob, Q = _n11_q4_fixture()
    β0 = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
          s1 = let μ = sum(prob.y1) / length(prob.y1)
              [log(sqrt(sum((prob.y1 .- μ) .^ 2) / max(length(prob.y1) - 1, 1)))]
          end,
          s2 = let μ = sum(prob.y2) / length(prob.y2)
              [log(sqrt(sum((prob.y2 .- μ) .^ 2) / max(length(prob.y2) - 1, 1)))]
          end,
          rho = [0.0])
    Λ0 = Matrix(Symmetric([0.32 0.04 0.02 0.0; 0.04 0.30 0.0 0.03; 0.02 0.0 0.14 0.01; 0.0 0.03 0.01 0.12]))
    θ = pack_theta(β0, Λ0)
    _, g, u0, _ = marginal_and_exact_grad(prob, Q, θ; n_newton = 35)
    H1 = lc_metric(prob, Q, θ, u0; h = 1e-5, n_newton = 20)
    H2 = lc_metric(prob, Q, θ, u0; h = 2e-5, n_newton = 20)
    @test size(H1) == (10, 10)
    @test size(H2) == (10, 10)
    @test maximum(abs, H1 - H1') ≤ 1e-10
    @test maximum(abs, H2 - H2') ≤ 1e-10
    ev1 = eigvals(Symmetric(H1))
    ev2 = eigvals(Symmetric(H2))
    @test all(ev1 .> 0)
    @test all(ev2 .> 0)
    @test LinearAlgebra.tr(H1) > 0 && LinearAlgebra.tr(H2) > 0
    @test norm(H1 - H2) / (norm(H1) + 1e-8) < 0.15
    lc_ix = (length(g) - 9):length(g)
    g_lc = g[lc_ix]
    d1 = -(H1 \ g_lc)
    d2 = -(H2 \ g_lc)
    @test dot(g_lc, d1) < 0
    @test dot(g_lc, d2) < 0
    α = 1e-4
    nll0 = marginal_nll(prob, Q, θ; u0 = u0, n_newton = 25)[1]
    θp = copy(θ); θp[lc_ix] .+= α .* d1
    nll1 = marginal_nll(prob, Q, θp; u0 = u0, n_newton = 25)[1]
    @test isfinite(nll0) && isfinite(nll1)
    @test nll1 ≤ nll0 + 1e-6
end

@testset "cov 1125n11: lc_metric is tied to the 7+10 θ layout" begin
    prob, Q = _n11_q4_fixture(p = 5, seed = 1125112)
    @test DRModels.theta_len(prob) == 17
    β0 = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
          s1 = [-0.2], s2 = [-0.25], rho = [0.0])
    Λ0 = Matrix(Symmetric(0.35I(4)))
    θ = pack_theta(β0, Λ0)
    @test length(θ) == 17
    _, _, u0, _ = marginal_and_exact_grad(prob, Q, θ; n_newton = 25)
    H = lc_metric(prob, Q, θ, u0; h = 1e-5, n_newton = 15)
    @test size(H) == (10, 10)
    @test all(eigvals(Symmetric(H)) .> 0)
    @test all(isfinite, H)
end

@testset "cov 1125n11: lc_metric ridge lifts tiny curvature" begin
    prob, Q = _n11_q4_fixture(p = 5, seed = 1125113)
    βflat = (mu1 = zeros(size(prob.X1, 2)), mu2 = zeros(size(prob.X2, 2)),
             s1 = zeros(size(prob.Xs1, 2)), s2 = zeros(size(prob.Xs2, 2)), rho = [0.0])
    Λid = Matrix(Symmetric(0.5I(4)))
    θ = pack_theta(βflat, Λid)
    _, u0, _, _ = marginal_nll(prob, Q, θ; n_newton = 20)
    H = lc_metric(prob, Q, θ, u0; h = 1e-5, n_newton = 15)
    λ = eigvals(Symmetric(H))
    @test minimum(λ) ≥ 1e-3 - 1e-6
    @test maximum(λ) / minimum(λ) < 1e8
end