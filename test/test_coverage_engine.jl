# test_coverage_engine.jl — deepen coverage of genuinely-untested exported engine
# entry points and constructor guard rails.
#
# A PRIOR coverage pass anchored fixef / unpack_theta / joint_nll / joint_grad /
# build_Huu, and the standing Q-gates already exercise marginal_and_exact_grad /
# marginal_nll (FD-vs-exact) and augmented_tree_precision / random_caterpillar_tree
# (test_qgate_fd_gradient.jl, test_two_structured_gaussian_sparse.jl). This file
# adds the gaps those leave:
#
#   1. fit_q4_sparse_tmb — the HEADLINE verified q=4 engine has NO end-to-end
#      recovery test in the wired suite (only an `isdefined` presence check in
#      runtests.jl); the e2e drivers live in non-wired poc orphans. Add a small
#      recover-the-truth fit.
#   2. marginal_nll / marginal_and_exact_grad RETURN-TUPLE CONTRACT + the
#      cross-consistency of the two entry points' NLL at a shared θ — a property
#      the FD gradient gate never checks (it only compares gradients).
#   3. The bivariate bf() constructor GUARD RAILS for meta_V / relmat / animal:
#      these markers are deliberately unsupported on the bivariate q=4 path and
#      must raise. No existing test asserts these documented rejections.
#
# All problems are SMALL (p = 8 q=4 PLSM; identical generator family to
# test_analytic_grad.jl / check_sparse_tmb.jl) and self-contained (no fixtures).

using DRModels
using Test, LinearAlgebra, SparseArrays, Random, Statistics

# --- canonical small q=4 PLSM generator (same as test_qgate_fd_gradient.jl) ----
function _make_q4_problem(p; seed = 7, branch_length = 0.2)
    Random.seed!(seed)
    n = p
    phy = random_balanced_tree(p; branch_length = branch_length)
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
    return prob, Q_cond, y1, y2, X1, X2, βt
end

@testset "engine + constructor coverage" begin

    @testset "fit_q4_sparse_tmb: end-to-end recover (p=8)" begin
        prob, Q_cond, y1, y2, X1, X2, βt = _make_q4_problem(8)
        β0 = (mu1 = X1 \ y1, mu2 = X2 \ y2,
              s1 = [log(std(y1 .- X1 * (X1 \ y1)))],
              s2 = [log(std(y2 .- X2 * (X2 \ y2)))], rho = [0.0])
        Λ0 = Matrix(0.3 * I(4))
        res = fit_q4_sparse_tmb(prob, Q_cond; β0 = β0, Λ0 = Λ0)

        # the fit returns a finite, self-consistent optimum (loglik == −nll)
        @test isfinite(res.nll)
        @test res.loglik ≈ -res.nll
        @test res.converged

        # Λ̂ is a valid 4×4 SPD covariance
        @test size(res.Λ) == (4, 4)
        @test isposdef(Symmetric(res.Λ))

        # fixed-effect intercepts/slopes recover the truth. p=8 is a small sample,
        # so tolerances are loose — this guards the fit PATH, not statistical
        # efficiency. The mean structure (μ slopes) is the least-confounded by the
        # phylo random field.
        @test res.β.mu1[2] ≈ βt.mu1[2] atol = 0.25      # μ1 slope on x
        @test res.β.mu2[2] ≈ βt.mu2[2] atol = 0.25      # μ2 slope on x
        @test all(isfinite, res.β.mu1)
        @test all(isfinite, res.β.mu2)
    end

    @testset "marginal_nll / marginal_and_exact_grad: return contract + consistency" begin
        prob, Q_cond, y1, y2, X1, X2, _ = _make_q4_problem(8)
        β = (mu1 = X1 \ y1, mu2 = X2 \ y2,
             s1 = [log(std(y1 .- X1 * (X1 \ y1)))],
             s2 = [log(std(y2 .- X2 * (X2 \ y2)))], rho = [0.0])
        Λ = Matrix(Symmetric([0.30 0.06 0.02 0.0; 0.06 0.28 0.0 0.03;
                              0.02 0.0 0.14 0.01; 0.0 0.03 0.01 0.16]))
        θ = pack_theta(β, Λ)
        nu = 4 * prob.n_total

        # marginal_nll -> (nll, û, ch_H, W); W is the WhitenedPrior (#857: the
        # engine no longer forms P = Q ⊗ Λ⁻¹). Λ = W.L W.L', W.Pw = Q ⊗ I.
        nll, û, chH, W = marginal_nll(prob, Q_cond, θ; n_newton = 120)
        @test isfinite(nll)
        @test length(û) == nu
        @test all(isfinite, û)
        @test W isa DRModels.WhitenedPrior
        @test issparse(W.Pw)
        @test size(W.Pw) == (nu, nu)
        @test isposdef(Symmetric(Matrix(W.Pw)))         # whitened prior is SPD
        @test W.L * W.L' ≈ Λ rtol = 1e-12
        # the unwhitened prior precision is still recoverable and SPD
        P = DRModels.prior_precision(Q_cond, inv(Symmetric(Λ)))
        @test isposdef(Symmetric(Matrix(P)))

        # marginal_and_exact_grad -> (nll, grad, û, ch_H); θ has 7 β + 10 lc = 17
        nllg, g, ûg, _ = marginal_and_exact_grad(prob, Q_cond, θ; n_newton = 120)
        @test length(g) == 17
        @test length(θ) == 17
        @test all(isfinite, g)
        @test length(ûg) == nu

        # the two entry points must report the SAME marginal NLL at the same θ
        # (the FD gradient gate never checks this — only gradient agreement).
        @test nllg ≈ nll rtol = 1e-6
    end

    @testset "bivariate bf(): meta_V / relmat / animal markers are rejected" begin
        # The bivariate q=4 constructor path supports only `phylo(1 | group)`
        # structured markers (gaussian_bivariate.jl). meta_V / relmat / animal on
        # the location/scale predictors are documented guard rails and must raise.
        Random.seed!(11)
        G = 12; m = 3; n = G * m
        species = repeat(1:G, inner = m)
        x = randn(n); v = abs.(randn(n)) .+ 0.5
        y1 = randn(n); y2 = randn(n)
        data = (; y1, y2, x, v, species)

        # meta_V on all four predictors -> the meta_V guard.
        @test_throws ErrorException drm(
            bf(mu1 = @formula(y1 ~ x + meta_V(v)),
               mu2 = @formula(y2 ~ x + meta_V(v)),
               sigma1 = @formula(sigma1 ~ 1 + meta_V(v)),
               sigma2 = @formula(sigma2 ~ 1 + meta_V(v)),
               rho12 = @formula(rho12 ~ 1)),
            Gaussian(); data = data)

        # relmat on all four predictors -> the "phylo-only" guard.
        @test_throws ErrorException drm(
            bf(mu1 = @formula(y1 ~ x + relmat(1 | species)),
               mu2 = @formula(y2 ~ x + relmat(1 | species)),
               sigma1 = @formula(sigma1 ~ 1 + relmat(1 | species)),
               sigma2 = @formula(sigma2 ~ 1 + relmat(1 | species)),
               rho12 = @formula(rho12 ~ 1)),
            Gaussian(); data = data)

        # animal on all four predictors -> the "phylo-only" guard.
        @test_throws ErrorException drm(
            bf(mu1 = @formula(y1 ~ x + animal(1 | species)),
               mu2 = @formula(y2 ~ x + animal(1 | species)),
               sigma1 = @formula(sigma1 ~ 1 + animal(1 | species)),
               sigma2 = @formula(sigma2 ~ 1 + animal(1 | species)),
               rho12 = @formula(rho12 ~ 1)),
            Gaussian(); data = data)
    end

end
