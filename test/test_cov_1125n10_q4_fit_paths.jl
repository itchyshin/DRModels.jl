# cov lane 1125n10 — shallow q4 ML / REML fit paths (fit_q4_sparse_tmb, fit_q4_reml, reml_objective_at).
using DRModels
using Test, LinearAlgebra, Random, SparseArrays

function _cov_n10_q4_fixture(; p::Int = 6, seed::Int = 112520)
    Random.seed!(seed)
    phy = DRModels.random_balanced_tree(p; branch_length = 0.2)
    keep = setdiff(1:phy.n_total, [phy.root_index])
    Q_cond = phy.Q_topology[keep, keep]
    pos = Dict(n => i for (i, n) in enumerate(keep))
    leaf_node = [pos[phy.leaf_indices[k]] for k in 1:p]
    n = p
    x = randn(n)
    X1 = hcat(ones(n), x)
    X2 = hcat(ones(n), 0.5 .* x)
    Xs1 = reshape(ones(n), n, 1)
    Xs2 = reshape(ones(n), n, 1)
    Xr = zeros(n, 0)
    y1 = randn(n)
    y2 = randn(n)
    prob = DRModels.AugProblem(phy, length(keep), p, leaf_node, y1, y2, X1, X2, Xs1, Xs2, Xr)
    return prob, Q_cond
end

@testset "cov 1125n10: fit_q4_sparse_tmb smoke" begin
    prob, Q = _cov_n10_q4_fixture()
    β0 = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
          s1 = [-0.5], s2 = [-0.5], rho = Float64[])
    Λ0 = Matrix(Symmetric(0.3I(4)))
    fit = DRModels.fit_q4_sparse_tmb(prob, Q; β0 = β0, Λ0 = Λ0, iterations = 40, g_tol = 1e-2, n_newton = 20)
    @test hasproperty(fit, :loglik)
    @test hasproperty(fit, :β)
    @test hasproperty(fit, :Λ)
    @test isfinite(fit.loglik)
    @test size(fit.Λ) == (4, 4)
    @test isposdef(Symmetric(fit.Λ))
    θ = DRModels.pack_theta(fit.β, fit.Λ)
    @test length(θ) == DRModels.theta_len(prob)
    nll, g, _, _ = DRModels.marginal_and_exact_grad(prob, Q, θ; n_newton = 15)
    @test isfinite(nll)
    @test all(isfinite, g)
    @test length(g) == length(θ)
end

@testset "cov 1125n10: fit_q4_reml and reml_objective_at" begin
    prob, Q = _cov_n10_q4_fixture(seed = 112521)
    β0 = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
          s1 = [-0.4], s2 = [-0.35], rho = Float64[])
    Λ0 = Matrix(Symmetric(0.28I(4)))
    reml = DRModels.fit_q4_reml(prob, Q; beta0 = β0, Lambda0 = Λ0,
                                iterations = 50, g_tol = 1e-2, n_newton = 20)
    @test hasproperty(reml, :reml_loglik)
    @test hasproperty(reml, :ml_loglik)
    @test hasproperty(reml, :phi)
    @test isfinite(reml.reml_loglik) || reml.reml_loglik == -Inf
    @test isfinite(reml.ml_loglik)
    @test length(reml.phi) == DRModels.phi_len(prob)
    @test reml.iterations isa Integer && reml.iterations >= 0
    obj = DRModels.reml_objective_at(prob, Q, reml.phi; beta0 = β0, u0 = reml.u_hat, n_newton = 20)
    @test hasproperty(obj, :reml_loglik)
    @test hasproperty(obj, :raw_reml_ll)
    @test hasproperty(obj, :beta)
    @test isfinite(obj.reml_loglik) || obj.reml_loglik == -Inf
    @test all(isfinite, obj.beta.mu1)
    @test all(isfinite, obj.beta.s2)
    @test length(obj.u) == 4 * prob.n_total
    if isfinite(reml.reml_loglik) && isfinite(reml.ml_loglik)
        @test reml.reml_loglik < reml.ml_loglik + 1e-6
    end
end

@testset "cov 1125n10: chol diagnostic hooks" begin
    prob, Q = _cov_n10_q4_fixture(p = 4, seed = 112522)
    β0 = (mu1 = prob.X1 \ prob.y1, mu2 = prob.X2 \ prob.y2,
          s1 = [-0.3], s2 = [-0.3], rho = Float64[])
    Λ0 = Matrix(Symmetric(0.3I(4)))
    DRModels.reset_chol_diagnostics!()
    @test DRModels.CHOL_FACTORIZATIONS[] == 0
    @test DRModels.CHOL_REUSE_FALLBACKS[] == 0
    cache = DRModels.CholPatternCache()
    @test cache.factor === nothing
    @test cache.pattern_colptr === nothing
    _ = DRModels.fit_q4_sparse_tmb(prob, Q; β0 = β0, Λ0 = Λ0, iterations = 20,
                                  g_tol = 1e-2, n_newton = 15)
    @test DRModels.CHOL_FACTORIZATIONS[] >= 1
end