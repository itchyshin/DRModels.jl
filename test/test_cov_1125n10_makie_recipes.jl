# cov lane 1125n10 — DRModelsMakieExt drawing recipes (ext/DRModelsMakieExt.jl); optional weakdeps.
using DRModels
using Test, Random

const _HAS_MAKIE = Base.find_package("CairoMakie") !== nothing &&
    Base.find_package("AlgebraOfGraphics") !== nothing

@testset "cov 1125n10: drm_figure recipe smoke" begin
    profile_tuple = (
        x = [-1.0, 0.0, 1.0],
        deviance = [4.0, 0.0, 3.5],
        estimate = 0.0,
        cutoff = 3.84,
        k = 2,
        param = :mu,
        coef = "x",
        level = 0.95,
    )
    surface_tuple = (
        x = [0.0, 0.5],
        y = [0.0, 0.5],
        z = [2.0 1.0; 1.0 0.0],
        k1 = 1,
        k2 = 2,
    )
    corpairs_tuple = (rho = [0.2, 0.25, 0.3], constant = false)
    if !_HAS_MAKIE
        @test isempty(methods(drm_figure))
        @test_skip false
    else
        using CairoMakie, AlgebraOfGraphics
        @test !isempty(methods(drm_figure))
        fig1 = drm_figure(profile_tuple; kind = :profile)
        @test fig1 isa CairoMakie.Figure
        fig2 = drm_figure(surface_tuple; kind = :parameter_surface)
        @test fig2 isa CairoMakie.Figure
        fig3 = drm_figure((rho = Float64[], constant = true); kind = :corpairs)
        @test fig3 isa CairoMakie.Figure
        fig4 = drm_figure((rho = [0.35], constant = true); kind = :corpairs)
        @test fig4 isa CairoMakie.Figure
        @test_throws ArgumentError drm_figure((a = 1.0, b = 2.0))
        @test_throws ArgumentError drm_figure(profile_tuple; kind = :unknown)
        Random.seed!(112550)
        n = 80
        x = randn(n)
        y = 0.2 .+ 0.9 .* x .+ 0.35 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        fig5 = plot_profile(fit, 2; npoints = 7, span = 1.5)
        @test fig5 isa CairoMakie.Figure
    end
end