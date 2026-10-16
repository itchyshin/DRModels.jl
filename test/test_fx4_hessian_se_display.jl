# FX4 (c): non-PD Hessian / unconverged fit must not print usable Wald SEs
# (GasolineYield sweep t8; drmTMB #1554 neighbour). `is_converged` is unchanged.
using DRModels
using Test
using Random

@testset "FX4 Hessian SE display (GasolineYield / non-PD)" begin
    Random.seed!(20261016)
    n = 40
    x = randn(n)
    y = 0.3 .+ 0.5 .* x .+ 0.35 .* randn(n)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    @test is_converged(fit)
    @test all(isfinite, DRModels._display_se(fit))

    @testset "NaN vcov (non-PD Hessian) prints NaN SEs, not Inf" begin
        bad = DRModels.DrmFit(
            fit.family, fit.blocks, fit.coefnames, fit.theta,
            fill(NaN, size(fit.vcov)),
            fit.loglik, fit.nobs, true, fit.means, fit.obs, fit.scales)
        @test is_converged(bad)   # optimiser flag unchanged (#1554)
        se = DRModels._display_se(bad)
        @test all(isnan, se)
        @test !any(isinf, se)
        ct = coeftable(bad)
        @test all(isnan, ct.cols[2])
        shown = sprint(show, "text/plain", bad)
        @test occursin("Wald SEs withheld", shown)
        @test occursin("NaN", shown)
    end

    @testset "unconverged fit with a finite Hessian does not print SEs" begin
        unc = DRModels.DrmFit(
            fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov,
            fit.loglik, fit.nobs, false, fit.means, fit.obs, fit.scales)
        @test !is_converged(unc)
        @test all(isnan, DRModels._display_se(unc))
        ct = coeftable(unc)
        @test all(isnan, ct.cols[2])
        shown = sprint(show, "text/plain", unc)
        @test occursin("Wald SEs withheld", shown)
    end
end
