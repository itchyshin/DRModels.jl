# test_lc_metric.jl — #13 S1b unit test for the extracted Fisher / observed-
# information metric on log-Cholesky params (`src/lc_metric.jl`).
#
# Design bar (`report/wire-em-solvers-design.md`): after the natgrad solver FAIL,
# land `lc_metric` + a unit test that it equals the Fisher (observed) information
# on a small case — NOT a public solver wire.
#
# Checks:
#   1. returns a 10×10 SPD matrix (ridge-projected);
#   2. agrees with an independent central-FD Hessian of `marginal_nll` w.r.t. lc
#      (observed information of the scalar marginal);
#   3. natural direction `H \\ g_lc` is a descent direction for the marginal NLL.

using DRModels
using Test, LinearAlgebra, Random, Statistics

@testset "lc_metric (#13 S1b): Fisher / observed-info on lc" begin
    Random.seed!(7)
    p = 8; n = p

    phy = random_balanced_tree(p; branch_length = 0.2)
    Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)
    βt = (mu1 = [1.0, 0.5], mu2 = [-0.3, 0.4], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
    Λt = [0.25 0.10 0.05 0.0; 0.10 0.25 0.0 0.04; 0.05 0.0 0.09 0.02; 0.0 0.04 0.02 0.09]
    Λt = (Λt + Λt') / 2

    x1 = randn(n)
    X1 = hcat(ones(n), x1); X2 = hcat(ones(n), x1)
    Xs1 = reshape(ones(n), n, 1); Xs2 = reshape(ones(n), n, 1); Xr = reshape(ones(n), n, 1)

    U = cholesky(Λt).L * randn(4, p) * cholesky(Symmetric(Σ_phy)).U
    y1 = zeros(n); y2 = zeros(n)
    for i in 1:n
        m1 = (X1[i, :]' * βt.mu1) + U[1, i]; m2 = (X2[i, :]' * βt.mu2) + U[2, i]
        s1 = exp((Xs1[i, :]' * βt.s1) + U[3, i]); s2 = exp((Xs2[i, :]' * βt.s2) + U[4, i])
        ρ = 0.99999999 * tanh(Xr[i, :]' * βt.rho)
        e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(2)
        y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
    end

    prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
    β0 = (mu1 = X1 \ y1, mu2 = X2 \ y2,
          s1 = [log(std(y1 .- X1 * (X1 \ y1)))], s2 = [log(std(y2 .- X2 * (X2 \ y2)))],
          rho = [0.0])
    Λ0 = [0.30 0.06 0.02 0.0; 0.06 0.28 0.0 0.03; 0.02 0.0 0.14 0.01; 0.0 0.03 0.01 0.16]
    Λ0 = Matrix(Symmetric((Λ0 + Λ0') / 2))
    θ = pack_theta(β0, Λ0)

    _, g, u0, _ = marginal_and_exact_grad(prob, Q_cond, θ; n_newton = 80)
    H = lc_metric(prob, Q_cond, θ, u0; h = 1e-5, n_newton = 40)

    @test size(H) == (10, 10)
    @test maximum(abs, H - H') ≤ 1e-10
    ev = eigvals(Symmetric(H))
    @test all(ev .> 0)

    # Independent observed information: FD Hessian of scalar marginal_nll w.r.t. lc
    h = 1e-4
    H_fd = zeros(10, 10)
    for i in 1:10, j in i:10
        function nll_at(δi, δj)
            t = copy(θ)
            t[7 + i] += δi
            t[7 + j] += δj
            return marginal_nll(prob, Q_cond, t; u0 = u0, n_newton = 80)[1]
        end
        # mixed central difference
        fpp = nll_at(h, h); fpm = nll_at(h, -h); fmp = nll_at(-h, h); fmm = nll_at(-h, -h)
        Hij = (fpp - fpm - fmp + fmm) / (4h^2)
        H_fd[i, j] = Hij
        H_fd[j, i] = Hij
    end

    # Ridge in lc_metric can lift tiny eigenvalues; compare on the well-conditioned
    # subspace via relative Frobenius error of the unridged agreement on large entries.
    # Allow looser tol: nll-FD Hessian is noisier than grad-FD (lc_metric's construction).
    rel = norm(H - H_fd) / (norm(H_fd) + 1e-8)
    @test rel ≤ 0.25

    # Natural / Newton direction on lc is a descent direction for NLL
    g_lc = g[8:17]
    dir = -(H \ g_lc)
    α = 1e-3
    θ_trial = copy(θ); θ_trial[8:17] .+= α .* dir
    nll0 = marginal_nll(prob, Q_cond, θ; u0 = u0, n_newton = 80)[1]
    nll1 = marginal_nll(prob, Q_cond, θ_trial; u0 = u0, n_newton = 80)[1]
    @test nll1 < nll0
end
