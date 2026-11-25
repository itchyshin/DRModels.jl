# cov lane 1125n6 — reml_q2.jl phi pack/unpack and admissibility gate.
using DRModels
using Test, LinearAlgebra

@testset "cov 1125n6: q2_reml_phi_len" begin
    @test DRModels.q2_reml_phi_len() == DRModels.lc_len(2) + 3
end

@testset "cov 1125n6: _q2_reml_pack_unpack roundtrip" begin
    Λ = [1.2 0.2; 0.2 0.9]
    σ = [1.5, 2.0]
    ρ = 0.35
    φ = DRModels._q2_reml_pack_phi(Λ, σ, ρ)
    @test length(φ) == DRModels.q2_reml_phi_len()
    Λ2, D, chΛ = DRModels._q2_reml_unpack_phi(φ)
    @test Λ2 ≈ Λ rtol = 1e-10
    @test D[1, 1] ≈ σ[1]^2 rtol = 1e-12
    @test D[2, 2] ≈ σ[2]^2 rtol = 1e-12
    @test chΛ isa Cholesky
    @test_throws ErrorException DRModels._q2_reml_pack_phi(Λ, [1.0], ρ)
end

@testset "cov 1125n6: _q2_reml_pack clamps rho" begin
    Λid = Matrix(1.0I, 2, 2)
    φ_hi = DRModels._q2_reml_pack_phi(Λid, [1.0, 1.0], 5.0)
    φ_lo = DRModels._q2_reml_pack_phi(Λid, [1.0, 1.0], -5.0)
    _, Dhi, _ = DRModels._q2_reml_unpack_phi(φ_hi)
    _, Dlo, _ = DRModels._q2_reml_unpack_phi(φ_lo)
    rhi = Dhi[1, 2] / sqrt(Dhi[1, 1] * Dhi[2, 2])
    rlo = Dlo[1, 2] / sqrt(Dlo[1, 1] * Dlo[2, 2])
    @test abs(rhi) <= 0.95 + 1e-10
    @test abs(rlo) <= 0.95 + 1e-10
end

@testset "cov 1125n6: _q2_lambda_admissible" begin
    @test DRModels._q2_lambda_admissible([2.0 0.1; 0.1 1.5])
    @test DRModels._q2_lambda_admissible([1.0 0.99; 0.99 1.0])
    @test !DRModels._q2_lambda_admissible([1.0 1.0; 1.0 1.0])
    @test !DRModels._q2_lambda_admissible([1.0 NaN; 0.0 1.0])
    @test !DRModels._q2_lambda_admissible([1.0 2.0; 2.0 1.0])
    modest = [1.0 0.01; 0.01 1.0]
    @test DRModels._q2_lambda_admissible(modest; maxcond = 1e12)
    @test !DRModels._q2_lambda_admissible(modest; maxcond = 1.0)
end