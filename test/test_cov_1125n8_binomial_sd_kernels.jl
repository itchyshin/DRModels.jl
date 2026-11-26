# cov lane 1125n8 — binomial SD regression kernels (src/binomial_sd_regression.jl).
using DRModels
using Test, LinearAlgebra, StatsModels

@testset "cov 1125n8: _binomial_stable_sech and _binomial_bounded_partial" begin
    for η in (-20.0, -1.0, 0.0, 1.0, 20.0)
        s = DRModels._binomial_stable_sech(η)
        @test s > 0 && isfinite(s)
        ρ, inn = DRModels._binomial_bounded_partial(η)
        @test -1 < ρ < 1
        @test inn > 0
    end
    ηbig = 15.0
    ρb, _ = DRModels._binomial_bounded_partial(ηbig)
    @test abs(ρb) < 1
end

@testset "cov 1125n8: partial-correlation Cholesky factor" begin
    η = [0.3, -0.2, 0.1]
    L = DRModels._binomial_partial_correlation_cholesky(η, 3)
    @test size(L) == (3, 3)
    R = L * L'
    @test issymmetric(R)
    @test all(diag(R) .≈ 1.0)
    pairs = DRModels._binomial_pairwise_correlations(η, 3)
    @test length(pairs) == 3
    @test pairs[1] ≈ R[1, 2] atol = 1e-12
    @test_throws ArgumentError DRModels._binomial_partial_correlation_cholesky([0.1], 3)
    @test_throws ArgumentError DRModels._binomial_partial_correlation_cholesky(η, 1)
end

@testset "cov 1125n8: _binomial_sd_scale" begin
    η = [-20.0, 0.0, 20.0]
    s = DRModels._binomial_sd_scale(η)
    @test all(isfinite, s)
    @test all(s .> 0)
    Zg = reshape([1.0, 0.0, 1.0], 3, 1)
    sg = DRModels._binomial_structured_sd_group_scale(Zg, [0.5])
    @test length(sg) == 3
end

@testset "cov 1125n8: _binomial_phylo_interaction_parts" begin
    fmu = bf(@formula(y ~ 1 + phylo_interaction(1 | host & pollinator)))
    parts = DRModels._binomial_phylo_interaction_parts(fmu.forms[1][2])
    @test parts.group1 == :host
    @test parts.group2 == :pollinator
    @test parts.slope === nothing
    f2 = bf(@formula(y ~ phylo_interaction(0 + x | host & pollinator)))
    p2 = DRModels._binomial_phylo_interaction_parts(f2.forms[1][2])
    @test p2.slope == :x
    @test_throws ArgumentError begin
        bf2 = bf(@formula(y ~ phylo_interaction(1 | host & host)))
        DRModels._binomial_phylo_interaction_parts(bf2.forms[1][2])
    end
end