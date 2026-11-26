# cov lane 1125n10 — visualization data providers + plot_* aliases (src/visualization.jl, plotting_ext.jl).
using DRModels
using Test, Random, StatsModels

@testset "cov 1125n10: profile and surface validation guards" begin
    Random.seed!(112540)
    n = 120
    x = randn(n)
    y = 0.3 .+ 0.7 .* x .+ 0.4 .* randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    @test_throws ArgumentError profile_curve(fit, 2; npoints = 2)
    @test_throws ArgumentError parameter_surface(fit, 1, 2; npoints = 1)
    bad = DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov,
                 fit.loglik, fit.nobs, fit.converged, fit.means, fit.obs, fit.scales,
                 fit.formula, nothing, fit.nllgrad, fit.ranef, fit.estim_method,
                 fit.reml_loglik, fit.ml_loglik, fit.marginal, fit.phylo_penalty,
                 fit.penalty, fit.iterations, fit.phylo_scale)
    @test_throws ArgumentError profile_curve(bad, 1)
    @test_throws ArgumentError parameter_surface(bad, 1, 2)
    curve = profile_curve(fit, 2; npoints = 7, span = 1.5)
    @test curve.level ≈ 0.95 atol = 1e-12
    @test curve.k == 2
    @test all(isfinite, curve.x)
    @test all(isfinite, curve.deviance)
    surf = parameter_surface(fit, 1, 2; npoints = 5, span = 1.5)
    @test surf.k1 == 1 && surf.k2 == 2
    @test all(isfinite, surf.z)
end

@testset "cov 1125n10: corpairs varying vs constant rho12" begin
    Random.seed!(112541)
    n = 80
    x = randn(n)
    z1 = randn(n)
    z2 = randn(n)
    e1 = 0.5 .* z1
    e2 = 0.5 .* (0.3 .* z1 .+ sqrt(1 - 0.3^2) .* z2)
    y = x .+ e1
    y2 = -0.2 .* x .+ e2
    const_fit = drm(
        bf(;
            mu1 = @formula(y ~ 1),
            mu2 = @formula(y2 ~ 1),
            sigma1 = @formula(sigma1 ~ 1),
            sigma2 = @formula(sigma2 ~ 1),
            rho12 = @formula(rho12 ~ 1),
        ),
        Gaussian();
        data = (; y, y2, x),
    )
    cd0 = corpairs_data(const_fit)
    @test cd0.constant
    @test length(cd0.rho) == n
    var_fit = drm(
        bf(;
            mu1 = @formula(y ~ 1),
            mu2 = @formula(y2 ~ 1),
            sigma1 = @formula(sigma1 ~ 1),
            sigma2 = @formula(sigma2 ~ 1),
            rho12 = @formula(rho12 ~ x),
        ),
        Gaussian();
        data = (; y, y2, x),
    )
    cd1 = corpairs_data(var_fit)
    @test !cd1.constant
    @test length(cd1.rho) == n
    @test maximum(abs.(cd1.rho .- cd1.rho[1])) > 0.01
end

@testset "cov 1125n10: plot_* aliases prepare then fail without Makie" begin
    Random.seed!(112542)
    n = 60
    x = randn(n)
    y = 0.5 .+ x .+ 0.3 .* randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    data = profile_curve(fit, 1; npoints = 5, span = 1.0)
    @test_throws MethodError plot_profile(fit, 1; npoints = 5, span = 1.0)
    @test_throws MethodError plot_parameter_surface(fit, 1, 2; npoints = 3, span = 1.0)
    @test_throws MethodError plot_corpairs(fit)
    @test_throws MethodError drm_figure(data; kind = :profile)
    @test occursin("x", data.coef) || data.param === :mu
end

@testset "cov 1125n10: _profile_plot_deviance sentinel path" begin
    @test isnan(DRModels._profile_plot_deviance(1e20, 0.0, "cov1125n10", "edge"))
    @test DRModels._profile_plot_deviance(3.0, 1.0, "cov1125n10", "ok") == 4.0
    @test_throws ArgumentError DRModels._profile_plot_deviance(
        0.0, 1.0, "cov1125n10", "below_mle",
    )
end