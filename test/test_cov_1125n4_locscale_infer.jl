# test_cov_1125n4_locscale_infer.jl — group-level summaries (src/locscale_infer.jl).
using DRModels
using Test, LinearAlgebra, Random

@testset "cov 1125n4: _ls_components geometry" begin
    for (a, b, ρ) in ((1.0, 1.0, 0.0), (4.0, 9.0, 0.2), (0.25, 0.36, -0.3))
        off = ρ * sqrt(a * b)
        Λ = [a off; off b]
        c = DRModels._ls_components(Λ)
        @test c.sd_mu ≈ sqrt(a)
        @test c.sd_psi ≈ sqrt(b)
        @test c.cor_mu_psi ≈ ρ atol = 1e-12
    end
    Λ0 = [2.0 0.0; 0.0 3.0]
    @test DRModels._ls_components(Λ0).cor_mu_psi == 0.0
end

@testset "cov 1125n4: _ls_se" begin
    @test DRModels._ls_se(nothing) === nothing
    V = [0.04 0.01; 0.01 0.09]
    se = DRModels._ls_se(V)
    @test se ≈ [0.2, 0.3] atol = 1e-12
    Vbad = [0.0 0.0; 0.0 1.0]
    @test any(isnan, DRModels._ls_se(Vbad))
end

@testset "cov 1125n4: _ls_components on engine-scale Λ" begin
    Λ4 = [0.25 0.05 0.02 0.0
          0.05 0.22 0.0 0.03
          0.02 0.0 0.18 0.01
          0.0 0.03 0.01 0.20]
    Λ4 = (Λ4 + Λ4') / 2
  @test isposdef(Symmetric(Λ4))
    for i in 1:4, j in i+1:4
        sub = Λ4[i:i+1, i:i+1]
        c = DRModels._ls_components(sub)
        ρ = sub[1, 2] / sqrt(sub[1, 1] * sub[2, 2])
        @test c.cor_mu_psi ≈ ρ atol = 1e-12
    end
end