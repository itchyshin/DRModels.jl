# cov lane 1125n6 — binomial_sd_regression.jl partial-correlation geometry.
using DRModels
using Test, LinearAlgebra

@testset "cov 1125n6: _binomial_stable_sech" begin
    for η in (-20.0, -5.0, 0.0, 5.0, 20.0, 50.0)
        s = DRModels._binomial_stable_sech(η)
        @test s > 0 && isfinite(s)
        t = sech(η)
        isfinite(t) && @test s ≈ t rtol = 1e-10
    end
    @test DRModels._binomial_stable_sech(1000.0) >= 0
end

@testset "cov 1125n6: _binomial_bounded_partial clamp" begin
    ρ, innov = DRModels._binomial_bounded_partial(0.0)
    @test abs(ρ) < 0.01
    @test innov > 0.9
    ρ2, _ = DRModels._binomial_bounded_partial(20.0)
    @test abs(ρ2) < 1.0
    ρ3, _ = DRModels._binomial_bounded_partial(-20.0)
    @test abs(ρ3) < 1.0
    @test sign(DRModels._binomial_bounded_partial(0.5)[1]) == sign(0.5)
end

@testset "cov 1125n6: _binomial_partial_correlation_cholesky" begin
    L2 = DRModels._binomial_partial_correlation_cholesky([0.4], 2)
    R2 = L2 * L2'
    @test R2[1, 1] ≈ 1.0
    @test R2[2, 2] ≈ 1.0
    @test abs(R2[1, 2]) < 0.5
    etas = [0.2, -0.3, 0.1]
    L3 = DRModels._binomial_partial_correlation_cholesky(etas, 3)
  @test size(L3) == (3, 3)
    @test all(diag(L3) .> 0)
    R3 = L3 * L3'
    @test issymmetric(R3)
    @test all(eigvals(Symmetric(R3)) .> 0)
    @test_throws ArgumentError DRModels._binomial_partial_correlation_cholesky([0.0], 1)
    @test_throws ArgumentError DRModels._binomial_partial_correlation_cholesky([0.0, 0.1], 3)
end

@testset "cov 1125n6: _binomial_pairwise_correlations" begin
    pw = DRModels._binomial_pairwise_correlations([0.0, 0.25, -0.1], 3)
    @test length(pw) == 3
    @test pw[1] < pw[2]
    @test all(abs.(pw) .< 1)
end

@testset "cov 1125n6: _binomial_partial_correlation_names" begin
    @test DRModels._binomial_partial_correlation_names(:site, 2) == ["site:atanh_rho"]
    n3 = DRModels._binomial_partial_correlation_names(:id, 3)
    @test length(n3) == 3
    @test all(startswith.(n3, "id:atanh"))
end

@testset "cov 1125n6: _binomial_sd_scale soft clamp" begin
    v = DRModels._binomial_sd_scale([-20.0, 0.0, 15.0])
    @test all(v .> 0)
    @test v[1] < exp(-10)
    @test v[3] < exp(20)
end