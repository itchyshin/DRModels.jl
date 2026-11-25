# cov lane 1125n6 — cumulative.jl cut/score helpers (no full drm fit).
using DRModels
using Test, Random, Statistics

@testset "cov 1125n6: _cumulative_cuts monotone" begin
    δ = [0.0, log(0.5), log(0.3), -0.2]
    cuts = DRModels._cumulative_cuts(δ)
    @test cuts[1] == δ[1]
    @test all(diff(cuts) .> 0)
    @test length(cuts) == length(δ)
    δ1 = [1.5]
    @test only(DRModels._cumulative_cuts(δ1)) == 1.5
end

@testset "cov 1125n6: _cumulative_loglik partition" begin
    δ = [0.0, 0.4, -0.1]
    cuts = DRModels._cumulative_cuts(δ)
    K, nc = length(cuts) + 1, length(cuts)
    η = 0.15
    logs = [DRModels._cumulative_loglik(k, η, cuts, K, nc) for k in 1:K]
    @test all(isfinite, logs)
    @test sum(exp, logs) ≈ 1.0 rtol = 1e-9
    @test maximum(logs) <= 0.0
end

@testset "cov 1125n6: _cumulative_score expectation" begin
    η = [-0.2, 0.3, 0.8]
    δ = [0.0, log(0.4)]
    cuts = DRModels._cumulative_cuts(δ)
    K = length(cuts) + 1
    sc = DRModels._cumulative_score(η, cuts, K)
    @test length(sc) == length(η)
    @test all(1 .<= sc .<= K)
    @test sc[2] > sc[1]
end

@testset "cov 1125n6: _cumulative_cut_init empirical" begin
    Random.seed!(1125)
    K = 4
    y = rand(1:K, 40)
    δ0 = DRModels._cumulative_cut_init(y, K, length(y))
    @test length(δ0) == K - 1
    @test all(isfinite, δ0)
    cuts = DRModels._cumulative_cuts(δ0)
    @test all(diff(cuts) .> 0)
end

@testset "cov 1125n6: _cumulative_marginal symbol" begin
    @test DRModels._cumulative_marginal(:LA) === :LA
    @test DRModels._cumulative_marginal(:Laplace) === :Laplace
    @test_throws ArgumentError DRModels._cumulative_marginal(:GHQ)
end

@testset "cov 1125n6: _cumulative_laplace_reject message" begin
    msg = try
        DRModels._cumulative_laplace_reject("random slopes")
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("random slopes", msg)
    @test occursin("Laplace", msg)
end