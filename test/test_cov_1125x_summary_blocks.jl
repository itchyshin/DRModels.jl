# test_cov_1125x_summary_blocks.jl — block labels and null notes (src/summary.jl).
using DRModels
using Test

@testset "summary block titles" begin
    @test DRModels._block_title(:mu) == "Mean model (μ)"
    @test DRModels._block_title(:rho12) == "Correlation (atanh ρ12)"
    @test DRModels._block_title(:temporal_phi) == "Temporal AR1 persistence (atanh φ)"
    @test DRModels._block_title(:custom_block) == "custom_block"
    @test DRModels._family_name(Gaussian()) == "Gaussian"
end

@testset "summary block null notes" begin
    mu_notes = DRModels._block_null_note(:mu)
    @test mu_notes == ("H0: coefficient = 0",)
    rho = DRModels._block_null_note(:rho12)
    @test first(rho) == "H0: rho12 = 0 (atanh scale)"
    sig = DRModels._block_null_note(:sigma)
    @test length(sig) == 2
    @test occursin("log σ", first(sig))
    @test DRModels._block_null_note(:cutpoints) == ()
end

@testset "DRMControl show" begin
    c = DRMControl(maxit = 500, multi_start = 3, se = false, optimizer = :bfgs)
    s = sprint(show, c)
    @test occursin("DRMControl", s)
    @test occursin("multi_start = 3", s)
    @test occursin(":bfgs", s)
end