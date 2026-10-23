# Visualization data providers: parameter_surface (2-D profile deviance grid)
# and corpairs_data (between-response correlation summary). Backend-free — we
# test the numbers a plot would consume, mirroring drmTMB's plot_* helpers.
using DRModels
using Test, Random

@testset "visualization data providers" begin
    @testset "profile_curve — 1-D profile diagnostic data" begin
        Random.seed!(20260604)
        n = 300
        x = randn(n)
        y = 0.2 .+ 0.8 .* x .+ 0.6 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data=(; y, x))
        curve = profile_curve(fit, 2; npoints=21, span=2.5)
        @test length(curve.x) == 21
        @test length(curve.deviance) == 21
        @test curve.param === :mu
        @test curve.coef == "x"
        @test curve.estimate in curve.x
        @test curve.cutoff > 0
        @test all(curve.deviance .>= -1e-8)
        @test minimum(curve.deviance) ≤ 1e-8
        @test curve.deviance[1] > 1.0 && curve.deviance[end] > 1.0
    end

    @testset "parameter_surface — 2-D profile deviance grid" begin
        Random.seed!(20260601)
        n = 300
        x = randn(n)
        y = 1.0 .+ 0.6 .* x .+ 0.5 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data=(; y, x))
        # profile the two mean coefficients (intercept = 1, slope = 2) over the σ nuisance
        surf = parameter_surface(fit, 1, 2; npoints=11, span=2.5)
        @test length(surf.x) == 11 && length(surf.y) == 11
        @test size(surf.z) == (11, 11)
        @test all(surf.z .>= -1e-6)            # profile deviance is non-negative
        @test minimum(surf.z) < 0.3            # ~0 near the MLE (grid straddles θ̂)
        # the estimate sits inside the grid span
        @test surf.x[1] < coef(fit)[1] < surf.x[end]
        @test surf.y[1] < coef(fit)[2] < surf.y[end]
        # corners (≈ span·se away in both) should be well above the χ²₂ 95% level
        @test surf.z[1, 1] > 1.0 && surf.z[end, end] > 1.0
    end

    @testset "parameter_surface — bad indices error" begin
        Random.seed!(20260602)
        n = 100
        x = randn(n)
        y = 0.5 .+ x .+ randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data=(; y, x))
        @test_throws ArgumentError parameter_surface(fit, 1, 1)     # not distinct
        @test_throws ArgumentError parameter_surface(fit, 1, 99)    # out of range
        @test_throws ArgumentError profile_curve(fit, 99)
    end

    @testset "corpairs_data — bivariate constant vs univariate empty" begin
        Random.seed!(20260603)
        n = 500
        x = randn(n)
        z1 = randn(n)
        z2 = randn(n)
        ρ = 0.4
        e1 = 0.6 .* z1
        e2 = 0.7 .* (ρ .* z1 .+ sqrt(1 - ρ^2) .* z2)
        y = 0.3 .+ 0.5 .* x .+ e1
        y2 = -0.1 .+ 0.2 .* x .+ e2
        bfit = drm(
            bf(;
                mu1=@formula(y ~ x),
                mu2=@formula(y2 ~ x),
                sigma1=@formula(sigma1 ~ 1),
                sigma2=@formula(sigma2 ~ 1),
                rho12=@formula(rho12 ~ 1)
            ),
            Gaussian();
            data=(; y, y2, x),
        )
        cd = corpairs_data(bfit)
        @test cd.constant
        @test length(cd.rho) == n
        @test cd.rho[1] ≈ ρ atol = 0.12

        ufit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data=(; y, x))
        @test isempty(corpairs_data(ufit).rho)
    end
end
