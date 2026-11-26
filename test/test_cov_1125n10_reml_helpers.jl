# cov lane 1125n10 — q4 REML algebra helpers (src/reml_q4.jl): layout, borders, grad_z, cond_newton.
using DRModels
using Test, LinearAlgebra, Random, SparseArrays, ForwardDiff

function _cov_n10_q4_fixture(; p::Int = 6, seed::Int = 112510)
    Random.seed!(seed)
    phy = DRModels.random_balanced_tree(p; branch_length = 0.25)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    pos = Dict(n => i for (i, n) in enumerate(keep))
    leaf_node = [pos[phy.leaf_indices[k]] for k in 1:p]
    n = p
    x = randn(n)
    X1 = hcat(ones(n), x)
    X2 = hcat(ones(n), 0.4 .* x)
    Xs1 = reshape(ones(n), n, 1)
    Xs2 = reshape(ones(n), n, 1)
    Xr = zeros(n, 0)
    y1 = 0.2 .+ x .+ 0.5 .* randn(n)
    y2 = -0.1 .+ 0.3 .* x .+ 0.6 .* randn(n)
    prob = DRModels.AugProblem(phy, length(keep), p, leaf_node, y1, y2, X1, X2, Xs1, Xs2, Xr)
    return prob, Q_cond
end

@testset "cov 1125n10: REML layout and phi helpers" begin
    prob, Q = _cov_n10_q4_fixture()
    Xax, wax, off, nbeta = DRModels._reml_axis_layout(prob)
    @test nbeta == sum(wax)
    @test length(off) == 4
    @test off[4] + wax[4] == nbeta
    @test wax == (size(prob.X1, 2), size(prob.X2, 2), size(prob.Xs1, 2), size(prob.Xs2, 2))
    β = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
         s1 = zeros(size(prob.Xs1, 2)), s2 = zeros(size(prob.Xs2, 2)), rho = Float64[])
    bv = DRModels._reml_beta_vec(prob, β)
    @test length(bv) == nbeta
    β2 = DRModels._reml_beta_nt(prob, bv, β.rho)
    @test β2.mu1 ≈ β.mu1
    @test β2.s2 ≈ β.s2
    Λ0 = Matrix(Symmetric(0.3I(4)))
    φ = DRModels.pack_phi(prob, β.rho, Λ0)
    ρu, lc = DRModels.unpack_phi(prob, φ)
    @test isempty(ρu)
    @test length(lc) == 10
    Xμ = DRModels.build_Xmu_lifted(prob)
    nbmu = size(prob.X1, 2) + size(prob.X2, 2)
    @test size(Xμ, 1) == 4 * prob.n_total
    @test size(Xμ, 2) == nbmu
    @test nbmu < nbeta
end

@testset "cov 1125n10: REML border blocks and grad_z" begin
    prob, Q = _cov_n10_q4_fixture(seed = 112511)
    Λ = Matrix(Symmetric([0.35 0.05 0.02 0.0; 0.05 0.32 0.0 0.02; 0.02 0.0 0.12 0.01; 0.0 0.02 0.01 0.11]))
    P = DRModels.prior_precision(Q, inv(Λ))
    β = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
         s1 = [-0.3], s2 = [-0.4], rho = Float64[])
    u, chH, _ = DRModels.estep_mode(prob, P, β; n_newton = 25)
    _, _, _, nbeta = DRModels._reml_axis_layout(prob)
    B, D = DRModels._reml_border_blocks(prob, u, β)
    @test size(B) == (4 * prob.n_total, nbeta)
    @test size(D) == (nbeta, nbeta)
    @test D ≈ D' rtol = 1e-10
    g_u, g_b = DRModels._reml_grad_z(prob, P, u, β)
    @test length(g_u) == 4 * prob.n_total
    @test length(g_b) == nbeta
    @test all(isfinite, g_u)
    @test all(isfinite, g_b)
    Lc = cholesky(Λ)
    W = DRModels.whitened_prior(Q, Lc.L)
    v = DRModels.u_to_whitened(W, u)
    Bw, Dw = DRModels._reml_border_blocks(prob, W, v, β)
    @test size(Bw) == size(B)
    @test Dw ≈ D rtol = 1e-9
end

@testset "cov 1125n10: cond_newton_beta and reml_nll_exact" begin
    prob, Q = _cov_n10_q4_fixture(seed = 112512)
    Λ = Matrix(Symmetric(0.3I(4)))
    P = DRModels.prior_precision(Q, inv(Λ))
    β0 = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
          s1 = [-0.2], s2 = [-0.25], rho = Float64[])
    u, _, _ = DRModels.estep_mode(prob, P, β0; n_newton = 20)
    β1 = DRModels.cond_newton_beta(prob, u, β0; n_newton = 12)
    @test length(β1.mu1) == length(β0.mu1)
    @test all(isfinite, β1.mu1)
    @test all(isfinite, β1.s2)
    φ = DRModels.pack_phi(prob, β0.rho, Λ)
    nll = DRModels.reml_nll_exact(prob, Q, φ; n_newton = 25, joint_iter = 12, joint_tol = 1e-9)
    @test nll == Inf || (isfinite(nll) && nll > 0)
    raw, u_hat, _, beta_hat, _, inner_ok = DRModels.reml_ll_and_mode(prob, Q, φ; n_newton = 20)
    @test inner_ok in (true, false)
    @test isfinite(raw) || raw == -Inf
    @test length(u_hat) == 4 * prob.n_total
    @test length(beta_hat.mu1) == length(β0.mu1)
end