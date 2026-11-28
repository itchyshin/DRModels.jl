# Round 3 — deeper src/joint_missing_finite.jl + joint_missing_finite_bridge.jl branches.
using DRModels, Test, ForwardDiff

@testset "cov 1125x3 finite joint row likelihood oracle" begin
    n, K = 5, 3
    y = Union{Missing, Float64}[0.2, missing, 0.8, -0.3, missing]
    x = Union{Missing, String}["low", "middle", missing, "high", missing]
    z = collect(-1.0:0.5:1.0)
    Xstate = zeros(n, K, 2)
    for i in 1:n, k in 1:K
        Xstate[i, k, 1] = 1.0
        Xstate[i, k, 2] = 0.3 * (k - 2) + 0.1 * z[i]
    end
    model = DRModels.prepared_joint_model(y, x, Xstate, ones(n, 1), reshape(z, n, 1);
        predictor = :ordinal, levels = ["low", "middle", "high"], variable = :sev,
        mu_names = ["a", "b"], sigma_names = ["(Intercept)"], predictor_names = ["z"],
        original_row = collect(10:14))
    theta = [0.0, 0.2, log(0.7), 0.5, -0.2, log(1.1)]
    rows = DRModels.prepared_joint_rowloglik(model, theta)
    @test length(rows) == n
    @test all(isfinite, rows)
    @test DRModels.prepared_joint_nll(model, theta) ≈ -sum(rows) atol = 1e-12
    g = ForwardDiff.gradient(t -> DRModels.prepared_joint_nll(model, t), theta)
    @test all(isfinite, g)
    mom = DRModels.prepared_joint_conditional_moments(model, theta)
    @test size(mom.probabilities) == (n, K)
    @test all(x -> 1.0 - 1e-10 <= x <= 1.0 + 1e-10, sum(mom.probabilities; dims = 2))
    @test all(isfinite, mom.mean)
    fitted = DRModels.fit_prepared_joint(model; g_tol = 1e-7)
    @test fitted.fit.converged
    @test isfinite(fitted.fit.loglik)
    sm = DRModels.joint_missing_summary(fitted)
    @test sm.predictor === :ordinal
    @test sm.variable == :sev
    imp = DRModels.imputed(fitted; rows = :missing)
    @test length(imp.estimate) == count(!, model.observed_x)
end

@testset "cov 1125x3 finite joint categorical masks" begin
    n, K = 6, 3
    y = Union{Missing, Float64}[0.0, missing, 1.0, -0.5, missing, 0.2]
    x = Union{Missing, String}["a", missing, "b", "c", missing, "a"]
    Xstate = zeros(n, K, 1)
    Xstate[:, :, 1] .= 1.0
    model = DRModels.prepared_joint_model(y, x, Xstate, ones(n, 1), ones(n, 1);
        predictor = :categorical, levels = ["a", "b", "c"], variable = :type,
        original_row = collect(1:n))
    theta = DRModels.prepared_joint_initial(model)
    @test isfinite(DRModels.prepared_joint_nll(model, theta))
    rows = DRModels.prepared_joint_rowloglik(model, theta)
    @test all(isfinite, rows)
    fit = DRModels.fit_prepared_joint(model; g_tol = 1e-6)
    @test fit.metadata.predictor === :categorical
    @test length(fit.metadata.levels) == K
    tab = DRModels.imputed(fit; rows = :all)
    @test length(tab.estimate) == n
    @test_throws ArgumentError DRModels.fit_prepared_joint(model; initial = [0.0], g_tol = 1e-6)
    @test_throws ArgumentError DRModels.fit_prepared_joint(model; g_tol = -1.0)
end

@testset "cov 1125x3 finite joint initial helper" begin
    n, K = 3, 2
    Xstate = zeros(n, K, 2)
    Xstate[:, :, 1] .= 1.0
    Xstate[:, :, 2] .= [-1.0, 0.0, 1.0]
    model = DRModels.prepared_joint_model(
        Union{Missing, Float64}[0.1, missing, -0.2],
        Union{Missing, String}["x1", missing, "x2"],
        Xstate, ones(n, 1), reshape(0.1:0.1:0.3, n, 1);
        predictor = :ordinal, levels = ["x1", "x2"], variable = :v,
        predictor_names = ["z"], original_row = [1, 2, 3])
    θ0 = DRModels.prepared_joint_initial(model)
    @test length(θ0) == DRModels._finite_joint_ntheta(model)
    @test all(isfinite, θ0)
    @test isfinite(DRModels.prepared_joint_nll(model, θ0))
end