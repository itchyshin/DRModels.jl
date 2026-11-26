# cov lane 1125n9 — summary printing helpers (src/summary.jl).
using DRModels
using Test, Random, Distributions, LinearAlgebra

@testset "cov 1125n9: _block_title" begin
    @test occursin("Mean", DRModels._block_title(:mu))
    @test occursin("ρ12", DRModels._block_title(:rho12))
    @test occursin("log σ", DRModels._block_title(:sigma))
    @test occursin("Cholesky", DRModels._block_title(:recov))
    @test DRModels._block_title(:custom_block) == "custom_block"
end

@testset "cov 1125n9: _block_null_note" begin
    μn = DRModels._block_null_note(:mu)
    @test !isempty(μn) && occursin("coefficient", first(μn))
    σn = DRModels._block_null_note(:sigma)
    @test length(σn) == 2
    @test DRModels._block_null_note(:recov) == ("H0: Cholesky entry = 0 (no single interpretable null)",)
    @test isempty(DRModels._block_null_note(:zi))
end

@testset "cov 1125n9: _family_name and deviance sentinel" begin
    @test DRModels._family_name(DRModels.Gaussian()) == "Gaussian"
    @test DRModels._family_name(DRModels.Poisson()) == "Poisson"
    fit = DrmFit(DRModels.Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]],
                 [0.0], [1.0;;], -1e18, 5, false,
                 Dict(:mu => ones(5)), Dict(:mu => ones(5)), Dict(:sigma => ones(5)))
    @test isnan(deviance(fit))
    ok = DrmFit(DRModels.Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]],
                [0.0], [1.0;;], -4.0, 5, true,
                Dict(:mu => ones(5)), Dict(:mu => ones(5)), Dict(:sigma => ones(5)))
    @test deviance(ok) ≈ 8.0 rtol = 1e-12
    @test dof_residual(ok) == 4
end

@testset "cov 1125n9: _marginal_note" begin
    fe = DrmFit(DRModels.Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]],
              [0.0], [1.0;;], -1.0, 3, true,
              Dict(:mu => ones(3)), Dict(:mu => ones(3)), Dict(:sigma => ones(3)),
              nothing, nothing, nothing, nothing, :ML, NaN, -1.0, :LA)
    @test DRModels._marginal_note(fe) === nothing
    re = DrmFit(DRModels.Poisson(), [:mu => 1:2, :resd => 3:3],
                [:mu => ["(Intercept)", "x"], :resd => ["g"]],
                [0.0, 0.0, -1.0], Matrix(0.01I, 3, 3), -2.0, 10, true,
                Dict(:mu => ones(10)), Dict(:mu => ones(10)), Dict(:sigma => ones(10)),
                nothing, nothing, nothing, (g = randn(4),), :ML, NaN, -2.0, :Laplace)
    note = DRModels._marginal_note(re)
    @test note !== nothing
    @test occursin("Laplace", note)
    va = DrmFit(DRModels.Poisson(), [:mu => 1:1], [:mu => ["(Intercept)"]],
                [0.0], [1.0;;], -1.0, 4, true,
                Dict(:mu => ones(4)), Dict(:mu => ones(4)), Dict(:sigma => ones(4)),
                nothing, nothing, nothing, (g = [1.0],), :ML, NaN, -1.0, :VA)
    @test occursin("variational", DRModels._marginal_note(va))
end

@testset "cov 1125n9: rho12 accessor" begin
    Random.seed!(9)
    fit = drm(bf(mu1 = @formula(y1 ~ 1), mu2 = @formula(y2 ~ 1),
                 sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                 rho12 = @formula(rho12 ~ 1)),
              DRModels.Gaussian(); data = (; y1 = randn(8), y2 = randn(8)))
    ρ = rho12(fit)
    @test length(ρ) == 8
    @test all(-1 .< ρ .< 1)
    uni = DrmFit(DRModels.Poisson(), [:mu => 1:1], [:mu => ["(Intercept)"]],
                 [0.0], [1.0;;], 0.0, 1, true,
                 Dict(:mu => [0.0]), Dict(:mu => [0.0]), Dict(:sigma => [1.0]))
    @test_throws ArgumentError rho12(uni)
end