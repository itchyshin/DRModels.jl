# cov lane 1125n12 — prepared joint missing-predictor validation and summary kernels.
using DRModels
using Test, LinearAlgebra, Random

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

@testset "cov 1125n12: _joint_float_or_missing and _joint_matrix" begin
    @test DRModels._joint_float_or_missing(1.5, "x") === 1.5
    @test DRModels._joint_float_or_missing(missing, "x") === missing
    @test_throws ArgumentError DRModels._joint_float_or_missing("a", "x")
    @test_throws ArgumentError DRModels._joint_float_or_missing(Inf, "x")
    @test_throws ArgumentError DRModels._joint_float_or_missing(NaN, "x")
    A = randn(5, 2)
    @test size(DRModels._joint_matrix(A, 5, "X")) == (5, 2)
    @test_throws ArgumentError DRModels._joint_matrix(A, 4, "X")
    @test_throws ArgumentError DRModels._joint_matrix(zeros(5, 0), 5, "X")
    @test_throws ArgumentError DRModels._joint_matrix(fill(NaN, 3, 1), 3, "X")
end

@testset "cov 1125n12: prepared_joint_model admission" begin
    y = Union{Missing,Float64}[1.0, missing, 2.0, missing]
    xg = Union{Missing,Float64}[0.5, missing, 1.0, missing]
    xb = Union{Missing,Float64}[1.0, missing, 0.0, 1.0]
    X = ones(4, 1)
    @test_throws ArgumentError DRModels.prepared_joint_model(y, xg[1:3], X, X, X; predictor = :gaussian)
    @test_throws ArgumentError DRModels.prepared_joint_model(y, xg, X, X, X; predictor = :nope)
    @test_throws ArgumentError DRModels.prepared_joint_model(y, [1.5, missing, 0.5, missing], X, X, X; predictor = :bernoulli)
    m = DRModels.prepared_joint_model(y, xb, X, X, X; predictor = :bernoulli, original_row = [10, 11, 12, 13])
    @test m.predictor === :bernoulli
    @test m.original_row == [10, 11, 12, 13]
    @test m.row_state[2] === :both_missing
    @test m.row_state[4] === :x_observed_y_missing
    @test_throws ArgumentError DRModels.prepared_joint_model(y, xg, X, X, X; predictor = :gaussian, mu_names = ["mi(x)"])
    @test_throws ArgumentError DRModels.prepared_joint_model(y, xg, X, X, X; predictor = :gaussian,
        mu_names = ["a", "b"], original_row = [1, 1, 2, 3])
end

@testset "cov 1125n12: prepared_joint_nll and fit_prepared_joint metadata" begin
    Random.seed!(11251201)
    n = 14
    z = randn(n)
    y = 0.2 .+ 0.5 .* z .+ 0.6 .* randn(n)
    x = Union{Missing,Float64}[i % 3 == 0 ? missing : 0.1 + 0.2 * z[i] for i in 1:n]
    X = hcat(ones(n), z)
    model = DRModels.prepared_joint_model(y, x, X, ones(n, 1), X; predictor = :gaussian)
    θ0 = DRModels.prepared_joint_initial(model)
    @test length(θ0) == DRModels._joint_ntheta(model)
    @test all(isfinite, θ0)
    rowll = DRModels.prepared_joint_rowloglik(model, θ0)
    @test length(rowll) == n
    @test all(isfinite, rowll[rowll .!= 0])
    @test DRModels.prepared_joint_nll(model, θ0) ≈ -sum(rowll) rtol = 1e-14
    fitted = DRModels.fit_prepared_joint(model; g_tol = 1e-4, initial = θ0)
    @test fitted.fit.loglik == -DRModels.prepared_joint_nll(model, fitted.fit.theta)
    summary = DRModels.joint_missing_summary(fitted)
    @test summary.predictor === :gaussian
    @test summary.all_rows == n
    @test summary.predictor_only_rows == count(==( :both_missing), model.row_state)
    @test summary.covariance_status isa Symbol
    @test summary.optimizer_status isa Symbol
    @test length(summary.conditional_mean) == n
    @test length(summary.conditional_status) == n
    @test_throws ArgumentError DRModels.fit_prepared_joint(model; initial = θ0[1:(end - 1)])
    @test_throws ArgumentError DRModels.fit_prepared_joint(model; initial = fill(Inf, length(θ0)))
    @test_throws ArgumentError DRModels.fit_prepared_joint(model; initial = θ0, g_tol = 0)
    badθ = copy(θ0)
    badθ[1] = NaN
    @test_throws ArgumentError DRModels._joint_parameters(model, badθ)
end

@testset "cov 1125n12: Bernoulli prepared rowloglik masks" begin
    y = Union{Missing,Float64}[1.0, missing, 0.0, missing]
    x = Union{Missing,Float64}[1.0, 0.0, missing, missing]
    X = ones(4, 1)
    model = DRModels.prepared_joint_model(y, x, X, X, X; predictor = :bernoulli)
    θ = [0.1, 0.2, -0.1, 0.3]
    rows = DRModels.prepared_joint_rowloglik(model, θ)
    @test rows[4] == 0.0
    @test all(isfinite, rows[rows .!= 0])
    mom = DRModels.prepared_joint_conditional_moments(model, θ)
    @test mom.status[1] === :observed
    @test mom.status[4] in (:predictor_only, :bernoulli_posterior)
end