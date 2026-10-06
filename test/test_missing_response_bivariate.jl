# test_missing_response_bivariate.jl — correctness oracle for issue #19
# (per-cell missing responses on the bivariate q=4 phylogenetic location-scale
# engine).
#
# Three guarantees, in increasing strength:
#
#  1. leaf_nll BRANCH FORMULAS. The masked leaf NLL must equal, branch-for-branch,
#     the correct observed-data Gaussian likelihood: both observed ⇒ bivariate;
#     exactly one ⇒ that response's UNIVARIATE marginal (the other σ and ρ drop
#     out — the observed-data likelihood, not a plug-in); neither ⇒ 0. And the
#     all-observed default (o1=o2=true) reproduces the bivariate term BIT-FOR-BIT.
#
#  2. FD-vs-EXACT GRADIENT ≤ 1e-6 WITH MISSING CELLS. This is the real oracle.
#     The per-cell mask is a CONSTANT w.r.t. (u, η), so it must propagate through
#     the entire exact O(p) implicit-function gradient (the ForwardDiff-of-
#     leaf_hess Jη/leaf_hess_du terms) and still match a central finite difference
#     of the TRUE masked marginal. A subtle masking bug in ANY of the four
#     exact-gradient call sites would break this. Mirrors test_qgate_fd_gradient.jl
#     exactly, with a mixed missing pattern (y1-only, y2-only, both-missing).
#
#  3. MAR RECOVERY. A larger problem with ~15% missing-at-random cells still
#     recovers the mean fixed effects to a loose tolerance and converges — the
#     end-to-end sanity check that the masked engine fits, not just differentiates.

using DRModels
using Test, LinearAlgebra, Random, Statistics

# Reach the un-exported leaf kernel + masked make_problem internals.
const _leaf_nll = DRModels.leaf_nll

@testset "#19 bivariate missing-response" begin

    @testset "leaf_nll branch formulas (observed-data likelihood)" begin
        u = [0.20, -0.10, 0.15, -0.05]
        η1, η2, ηs1, ηs2, ηr = 0.5, -0.3, -0.4, -0.6, 0.4
        y1, y2 = 1.1, -0.7
        mu1 = η1 + u[1]; mu2 = η2 + u[2]
        s1 = exp(ηs1 + u[3]); s2 = exp(ηs2 + u[4])
        ρ = 0.99999999 * tanh(ηr)

        # both observed ⇒ bivariate Gaussian NLL (hand-coded reference)
        e1 = y1 - mu1; e2 = y2 - mu2
        quad = (e1^2/s1^2 - 2ρ*e1*e2/(s1*s2) + e2^2/s2^2) / (1 - ρ^2)
        biv_ref = log(2π) + 0.5*log(s1^2*s2^2*(1-ρ^2)) + 0.5*quad
        @test _leaf_nll(u, y1, y2, η1, η2, ηs1, ηs2, ηr, true, true) ≈ biv_ref atol=1e-12

        # default args ⇒ all-observed bivariate, BIT-FOR-BIT
        @test _leaf_nll(u, y1, y2, η1, η2, ηs1, ηs2, ηr) ===
              _leaf_nll(u, y1, y2, η1, η2, ηs1, ηs2, ηr, true, true)

        # y1 observed, y2 missing ⇒ univariate N(mu1, σ1²); σ2, ρ drop out
        uni1_ref = 0.5*log(2π) + log(s1) + 0.5*e1^2/s1^2
        @test _leaf_nll(u, y1, NaN, η1, η2, ηs1, ηs2, ηr, true, false) ≈ uni1_ref atol=1e-12
        # ...and the missing value is NEVER referenced: NaN in y2 does not leak
        @test isfinite(_leaf_nll(u, y1, NaN, η1, η2, ηs1, ηs2, ηr, true, false))

        # y2 observed, y1 missing ⇒ univariate N(mu2, σ2²)
        uni2_ref = 0.5*log(2π) + log(s2) + 0.5*e2^2/s2^2
        @test _leaf_nll(u, NaN, y2, η1, η2, ηs1, ηs2, ηr, false, true) ≈ uni2_ref atol=1e-12
        @test isfinite(_leaf_nll(u, NaN, y2, η1, η2, ηs1, ηs2, ηr, false, true))

        # neither observed ⇒ 0 (tip couples only through the y-independent prior)
        @test _leaf_nll(u, NaN, NaN, η1, η2, ηs1, ηs2, ηr, false, false) == 0.0
    end

    @testset "FD-vs-exact gradient ≤ 1e-6 WITH missing cells" begin
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

        # mixed missing pattern exercising all three masked branches:
        y1[3] = NaN          # leaf 3: y1 missing  (o1=false, o2=true)
        y2[5] = NaN          # leaf 5: y2 missing  (o1=true,  o2=false)
        y1[7] = NaN; y2[7] = NaN   # leaf 7: both missing (neither)

        prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
        @test prob.obs1 == [true, true, false, true, true, true, false, true]
        @test prob.obs2 == [true, true, true, true, false, true, false, true]

        # θ0: OLS starts from OBSERVED rows (X \ y would be NaN), non-diagonal Λ
        o1 = prob.obs1; o2 = prob.obs2
        bm1 = X1[o1, :] \ y1[o1]; bm2 = X2[o2, :] \ y2[o2]
        β0 = (mu1 = bm1, mu2 = bm2,
              s1 = [log(std(y1[o1] .- X1[o1, :] * bm1))],
              s2 = [log(std(y2[o2] .- X2[o2, :] * bm2))], rho = [0.0])
        Λ0 = [0.30 0.06 0.02 0.0; 0.06 0.28 0.0 0.03; 0.02 0.0 0.14 0.01; 0.0 0.03 0.01 0.16]
        Λ0 = Matrix(Symmetric((Λ0 + Λ0') / 2))
        θ = pack_theta(β0, Λ0)

        _, g_analytic, u_ref, _ = marginal_and_exact_grad(prob, Q_cond, θ; n_newton = 120)

        mnll(t) = marginal_nll(prob, Q_cond, Vector{Float64}(t); u0 = u_ref, n_newton = 120)[1]
        h = 1e-4
        g_fd = similar(θ)
        for k in eachindex(θ)
            tp = copy(θ); tp[k] += h
            tm = copy(θ); tm[k] -= h
            g_fd[k] = (mnll(tp) - mnll(tm)) / (2h)
        end

        @test maximum(abs, g_analytic .- g_fd) ≤ 1e-6
    end

    @testset "MAR recovery (mean fixed effects)" begin
        Random.seed!(20)
        p = 60; n = p
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

        # ~15% MAR: knock out random cells, but never both at the same leaf and
        # never so many that a mean coefficient loses identifiability.
        miss1 = randperm(MersenneTwister(1), n)[1:round(Int, 0.15n)]
        miss2 = randperm(MersenneTwister(2), n)[1:round(Int, 0.15n)]
        y1[miss1] .= NaN
        y2[setdiff(miss2, miss1)] .= NaN

        prob, Q_cond = make_problem(phy, y1, y2, X1, X2, Xs1, Xs2, Xr)
        o1 = prob.obs1; o2 = prob.obs2
        @test count(o1) >= size(X1, 2) && count(o2) >= size(X2, 2)

        bm1 = X1[o1, :] \ y1[o1]; bm2 = X2[o2, :] \ y2[o2]
        β0 = (mu1 = bm1, mu2 = bm2,
              s1 = [log(std(y1[o1] .- X1[o1, :] * bm1))],
              s2 = [log(std(y2[o2] .- X2[o2, :] * bm2))], rho = [0.0])
        Λ0 = Matrix(Symmetric(0.2I(4) + 0.02 * (ones(4, 4) - I(4))))
        θ0 = pack_theta(β0, Λ0)

        fit = fit_q4_sparse_tmb(prob, Q_cond; θ0 = θ0, n_newton = 60)
        # loose recovery of the mean intercepts/slopes (the cleanly-identified part)
        @test isapprox(fit.β.mu1, βt.mu1; atol = 0.6)
        @test isapprox(fit.β.mu2, βt.mu2; atol = 0.6)
    end

    @testset "drm() front-end fits with missing cells (end-to-end)" begin
        Random.seed!(31)
        p = 12; nrep = 4
        phy = random_balanced_tree(p; branch_length = 0.2)
        Σ_phy = sigma_phy_dense(phy; σ²_phy = 1.0)
        βt = (mu1 = [1.0, 0.5], mu2 = [-0.3, 0.4], s1 = [-0.4], s2 = [-0.5], rho = [0.3])
        Λt = [0.25 0.10 0.05 0.0; 0.10 0.25 0.0 0.04; 0.05 0.0 0.09 0.02; 0.0 0.04 0.02 0.09]
        Λt = (Λt + Λt') / 2
        U = cholesky(Λt).L * randn(4, p) * cholesky(Symmetric(Σ_phy)).U   # 4×p per-leaf REs

        sp_idx = repeat(1:p, inner = nrep); n = length(sp_idx)
        x = randn(n)
        y1 = zeros(n); y2 = zeros(n)
        for i in 1:n
            k = sp_idx[i]; u = U[:, k]
            m1 = βt.mu1[1] + βt.mu1[2]*x[i] + u[1]; m2 = βt.mu2[1] + βt.mu2[2]*x[i] + u[2]
            s1 = exp(βt.s1[1] + u[3]); s2 = exp(βt.s2[1] + u[4])
            ρ = 0.99999999 * tanh(βt.rho[1])
            e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(2)
            y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
        end
        species = phy.leaf_names[sp_idx]
        # knock out a handful of cells (NaN), keeping plenty observed per species
        y1[3] = NaN; y2[7] = NaN; y1[15] = NaN; y2[15] = NaN; y1[28] = NaN
        data = (; y1, y2, x, species)
        form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
                  mu2 = @formula(y2 ~ x + phylo(1 | species)),
                  sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                  sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                  rho12 = @formula(rho12 ~ 1))
        fit = drm(form, Gaussian(); data = data, tree = phy,
                  q4_iterations = 200, q4_n_newton = 30, q4_vcov = false)
        @test isfinite(loglik(fit))
        @test loglik(fit) < 0                       # determined fit, not a positive overfit
        @test size(fit.ranef.Sigma_a) == (4, 4)
    end
end
