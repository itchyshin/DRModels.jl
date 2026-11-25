# test_cov_1125n4_boundary_blup.jl — BLUP RMS + quiet switch (src/boundary_diagnostics.jl).
using DRModels
using Test, Random, Logging

@testset "cov 1125n4: _blup_rms" begin
    @test DRModels._blup_rms([1.0, -2.0, 3.0]) ≈ sqrt((1 + 4 + 9) / 3)
    @test DRModels._blup_rms(Float64[]) === nothing
    @test DRModels._blup_rms([1.0, NaN]) === nothing
    @test DRModels._blup_rms([1.0, Inf]) === nothing
    @test DRModels._blup_rms(nothing) === nothing
    @test DRModels._blup_rms(Dict(:a => 1)) === nothing
    @test DRModels._blup_rms(ones(1)) == 1.0
end

@testset "cov 1125n4: _without_boundary_warnings" begin
    val = DRModels._without_boundary_warnings() do
        get(task_local_storage(), :drm_quiet_boundary, false)
    end
    @test val == true
    # nested restore
    tls = task_local_storage()
    old = get(tls, :drm_quiet_boundary, false)
    DRModels._without_boundary_warnings() do
        @test get(task_local_storage(), :drm_quiet_boundary, false)
    end
    @test get(task_local_storage(), :drm_quiet_boundary, false) == old
end

@testset "cov 1125n4: locscale _ls_components" begin
    Λ = [4.0 1.2; 1.2 9.0]
    c = DRModels._ls_components(Λ)
    @test c.sd_mu ≈ 2.0
    @test c.sd_psi ≈ 3.0
    @test c.cor_mu_psi ≈ 1.2 / 6.0
    Λ2 = [0.25 0.0; 0.0 0.16]
    c2 = DRModels._ls_components(Λ2)
    @test c2.cor_mu_psi == 0.0
end

@testset "cov 1125n4: _variance_boundary returns nothing without structure" begin
    Random.seed!(1125)
    n = 80
    x = randn(n)
    y = 0.2 .+ x .+ randn(n)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    @test DRModels._variance_boundary(fit) === nothing
end