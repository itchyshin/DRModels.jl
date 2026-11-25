# test_cov_1125n4_fisherz_chol_rows.jl — fz_R_chol row geometry (src/fisherz_q4.jl).
using DRModels
using Test, LinearAlgebra, Random

@testset "cov 1125n4: fz_R_chol unit rows" begin
    Random.seed!(1125)
    for _ in 1:40
        θ = 2 .* randn(6) .- 0.5
        C = DRModels.fz_R_chol(θ)
        @test size(C) == (4, 4)
        @test tril(C, -1) == tril(C, -1)  # lower pattern
        for i in 1:4
            row = view(C, i, 1:i)
            @test norm(row) ≈ 1.0 atol = 1e-12
        end
        R = C * C'
        @test all(≈(1.0; atol = 1e-12), diag(R))
        @test issymmetric(R)
    end
end

@testset "cov 1125n4: fz_R angles at 0" begin
    z = zeros(6)
    R0 = fz_R(z)
    @test R0[2, 1] ≈ 0 atol = 1e-12
    @test R0[1, 2] ≈ R0[2, 1]
    for p in ((2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3))
        i, j = p
        @test abs(R0[i, j]) < 1
    end
end

@testset "cov 1125n4: fz_DRD diagonal matches exp d" begin
    φ = vcat([0.0, -0.5, 0.3, 0.1], randn(6))
    Σ = fz_DRD(φ)
    sd = exp.(φ[1:4])
    @test diag(Σ) ≈ sd .^ 2 rtol = 1e-12
    @test isposdef(Symmetric(Σ))
end

@testset "cov 1125n4: fz_correlations length" begin
    φ = vcat(zeros(4), zeros(6))
    c = fz_correlations(φ)
    @test length(c) == 6
    R = fz_R(φ[5:10])
    @test c[1] ≈ R[2, 1] atol = 1e-12
end