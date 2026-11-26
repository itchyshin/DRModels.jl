# Round 3 — src/gaussian_multi_re_sparse.jl, gaussian_sparse_lss.jl, joint_missing_finite.jl.
using DRModels, Test, Random, LinearAlgebra, SparseArrays

@testset "cov 1125x3 multi_re_sparse kernel smoke" begin
    rng = MersenneTwister(112507)
    n, G, H = 60, 6, 5
    g = rand(rng, 1:G, n)
    h = rand(rng, 1:H, n)
    x = randn(rng, n)
    y = 1.0 .+ 0.4 .* x .+ 0.5 .* randn(rng, G)[g] .+ 0.35 .* randn(rng, H)[h] .+ 0.15 .* randn(rng, n)
    comps = [(ones(n), g, G, "g"), (ones(n), h, H, "h")]
    Xμ = hcat(ones(n), x)
    Xσ = ones(n, 1)
    fit = DRModels._fit_multi_ranef_gaussian(Gaussian(), y, Xμ, Xσ, comps,
        ["(Intercept)", "x"], ["(Intercept)"], 1e-6)
    @test is_converged(fit)
    @test isfinite(loglik(fit))
    θ = fit.theta
    @test isfinite(fit.nll(θ))
    Gvec = zeros(length(θ))
    fit.nllgrad(Gvec, θ)
    @test maximum(abs, Gvec) < 1e-4
    des = DRModels._multi_re_design(comps, n)
    σcol = fill(0.5, size(des.Z, 2))
    Z̃s = des.Z * Diagonal(σcol)
    fac = DRModels._multi_re_factor(Z̃s, fill(0.5, n), y; allow_chol = true)
    @test fac.logdetM > 0
    q = size(Z̃s, 2)
    S = DRModels._multi_re_selinv(fac, des.Mpat)
    @test length(DRModels._selinv_diag(S, q)) == q
    @test all(isfinite, DRModels._selinv_diag(S, q))
end

@testset "cov 1125x3 sparse lss public tiny tree" begin
    phy = random_balanced_tree(8; branch_length = 0.3)
    species = repeat(phy.leaf_names, inner = 2)
    n = length(species)
    x = randn(n)
    y = 0.5 .+ 0.3 .* x .+ 0.2 .* randn(n)
    dat = (; y, x, species = String.(species))
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1),
           @formula(sd(species, phylogenetic) ~ 1))
    fit = drm(f, Gaussian(); data = dat, tree = phy, algorithm = :sparse, method = :ML,
              g_tol = 1e-5)
    @test isfinite(loglik(fit))
    @test length(coef(fit)) >= 3
    pred = predict(fit, dat)
    @test length(pred) == n
    @test all(isfinite, pred)
end

@testset "cov 1125x3 joint_missing_finite engine" begin
    model, theta = let
        n, K = 4, 3
        y = Union{Missing, Float64}[0.75, -0.45, missing, missing]
        x = Union{Missing, String}["middle", missing, "high", missing]
        z = [-0.7, 0.25, 0.8, -0.35]
        Xstate = zeros(n, K, 2)
        score = [-0.8, 0.15, 1.1]
        for i in 1:n, k in 1:K
            Xstate[i, k, 1] = 1.0
            Xstate[i, k, 2] = score[k] + 0.12 * z[i]
        end
        model = DRModels.prepared_joint_model(y, x, Xstate, ones(n, 1), reshape(z, n, 1);
            predictor = :ordinal, levels = ["low", "middle", "high"], variable = :severity,
            mu_names = ["(Intercept)", "state_score"], sigma_names = ["(Intercept)"],
            predictor_names = ["z"], original_row = [31, 7, 88, 19])
        theta = [0.2, 0.65, log(0.85), 0.45, -0.35, log(1.4)]
        model, theta
    end
    nll0 = DRModels.prepared_joint_nll(model, theta)
    @test isfinite(nll0)
    using ForwardDiff
    g = ForwardDiff.gradient(t -> DRModels.prepared_joint_nll(model, t), theta)
    @test length(g) == length(theta)
    @test all(isfinite, g)
    fitted = DRModels.fit_prepared_joint(model; g_tol = 1e-7)
    @test fitted.fit.converged || isfinite(fitted.fit.loglik)
    summary = DRModels.joint_missing_summary(fitted)
    @test isfinite(fitted.fit.loglik)
    @test size(summary.conditional_probabilities, 1) == length(model.original_row)
    imp = DRModels.imputed(fitted; rows = :all)
    @test hasproperty(imp, :estimate)
    @test length(imp.estimate) == length(model.y)
    catm, catt = let
        n, K = 6, 3
        y = Union{Missing, Float64}[0.1, missing, 0.5, -0.2, missing, 0.3]
        x = Union{Missing, String}["low", "middle", missing, "high", missing, "low"]
        Xstate = zeros(n, K, 2)
        for i in 1:n, k in 1:K
            Xstate[i, k, 1] = 1.0
            Xstate[i, k, 2] = 0.2 * k
        end
        model = DRModels.prepared_joint_model(y, x, Xstate, ones(n, 1), ones(n, 1);
            predictor = :categorical, levels = ["low", "middle", "high"], variable = :habitat,
            original_row = collect(1:n))
        theta = [0.0, 0.1, log(0.9), -0.2, 0.3]
        model, theta
    end
    @test isfinite(DRModels.prepared_joint_nll(catm, catt))
    fitc = DRModels.fit_prepared_joint(catm; g_tol = 1e-5)
    @test fitc.fit.converged
    @test all(isfinite, fitc.metadata.conditional_probabilities)
end