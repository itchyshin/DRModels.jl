# cov lane 1125n6 — joint_missing_finite.jl micro-kernels (no full drm fit).
using DRModels
using Test, LinearAlgebra, Random

@testset "cov 1125n6: _finite_joint matrix guards" begin
    @test DRModels._finite_joint_matrix(Matrix(1.0I, 3, 3), 3, "X") ≈ I
    @test_throws ArgumentError DRModels._finite_joint_matrix(zeros(2, 1), 3, "X")
    @test_throws ArgumentError DRModels._finite_joint_matrix(zeros(3, 0), 3, "X")
    @test_throws ArgumentError DRModels._finite_joint_matrix([1 Inf; 2 3], 2, "X")
    A = randn(4, 3, 2)
    B = DRModels._finite_joint_state_array(A, 4, 3)
    @test size(B) == (4, 3, 2)
    @test_throws ArgumentError DRModels._finite_joint_state_array(zeros(3, 2, 0), 3, 2)
end

@testset "cov 1125n6: _finite_joint_levels and values" begin
    @test DRModels._finite_joint_levels(["a", "b"]) == ["a", "b"]
    @test_throws ArgumentError DRModels._finite_joint_levels(["only"])
    @test_throws ArgumentError DRModels._finite_joint_levels(["a", "a"])
    labels = ["low", "high"]
    @test DRModels._finite_joint_value("low", labels, "x") == 1
    @test DRModels._finite_joint_value(missing, labels, "x") === missing
    @test_throws ArgumentError DRModels._finite_joint_value("nope", labels, "x")
    @test_throws ArgumentError DRModels._finite_joint_value(NaN, labels, "x")
end

@testset "cov 1125n6: _finite_joint_log1mexp" begin
    for x in [-5.0, -0.1, -0.8]
        ref = log1p(-exp(x))
        @test DRModels._finite_joint_log1mexp(x) ≈ ref rtol = 1e-14
    end
    # Branch uses `log(-expm1(x))` for x ≥ -log(2); throws for small positive x.
    @test_broken isfinite(DRModels._finite_joint_log1mexp(0.2)) # TODO: branch guard for x > 0
end

@testset "cov 1125n6: _finite_joint_ordinal_logprobabilities" begin
    η = 0.2
    cutraw = [0.0, log(0.5)]
    lp = DRModels._finite_joint_ordinal_logprobabilities(η, cutraw)
    @test length(lp) == 3
    @test all(isfinite, lp)
    @test sum(exp, lp) ≈ 1.0 rtol = 1e-10
    lp2 = DRModels._finite_joint_ordinal_logprobabilities(η, [-36.0, 10.0])
    @test all(isfinite, lp2)
end

@testset "cov 1125n6: _finite_joint_logsumexp" begin
    @test DRModels._finite_joint_logsumexp([0.0, 0.0]) ≈ log(2.0) rtol = 1e-14
    @test DRModels._finite_joint_logsumexp([-1000.0, -1001.0]) ≈ -1000.0 + log(1 + exp(-1)) rtol = 1e-10
    big = DRModels._finite_joint_logsumexp([500.0, 500.5, 499.0])
    @test isfinite(big)
end

@testset "cov 1125n6: prepared_joint_model tiny ordinal" begin
    n, K = 6, 3
    y = fill(0.5, n)
    x = fill("a", n)
    xs = collect(range(-0.5, 0.5; length = n))
    Xstate = reshape(xs, n, 1, 1)
    Xstate = repeat(Xstate, 1, K, 1)
    Xσ = fill(1.0, n, 1)
    model = DRModels.prepared_joint_model(y, x, Xstate, Xσ, zeros(n, 0);
                                         predictor = :ordinal, levels = ["a", "b", "c"],
                                         variable = :x, mu_names = ["b0"],
                                         sigma_names = ["logσ"])
    @test model.predictor === :ordinal
    θ = zeros(DRModels._finite_joint_ntheta(model))
    β, δ, α, cut = DRModels._finite_joint_parameters(model, θ)
    @test length(β) == 1 && length(δ) == 1
    means = DRModels._finite_joint_state_means(model, β)
    @test size(means) == (n, K)
    lp = DRModels._finite_joint_logprobabilities(model, α, cut)
    @test size(lp) == (n, K)
    @test all(isfinite, lp)
end