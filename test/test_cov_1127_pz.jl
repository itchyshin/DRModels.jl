# Coverage slice (2026-11-27): branches in src/r2.jl, src/visualization.jl,
# src/plotting_ext.jl, and src/rescale_response.jl that are cheap to pin and
# user-facing (refusal messages, plot-alias wiring, rescale eligibility).
using DRModels
using Test
using Random
using Statistics

@testset "cursor coverage 2026-11-27 (src p–z)" begin

    @testset "r2_constant_sigma — extra refusal and edge cases" begin
        Random.seed!(20261127)
        n = 80
        x = randn(n)

        @testset "bivariate fit is refused by name" begin
            y1 = randn(n)
            y2 = randn(n)
            fit = drm(
                bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                   sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                   rho12 = @formula(rho12 ~ 1)),
                Gaussian();
                data = (; y1, y2, x),
            )
            @test_throws ArgumentError r2_constant_sigma(fit)
            msg = try
                r2_constant_sigma(fit)
            catch e
                sprint(showerror, e)
            end
            @test occursin("bivariate", lowercase(msg))
        end

        @testset "meta_V marker is refused without a silent R²" begin
            yv = randn(n)
            vi = fill(0.25, n)
            f_meta = drm(
                bf(@formula(yv ~ x + meta_V(vi)), @formula(sigma ~ 1)),
                Gaussian();
                data = (; yv, x, vi),
            )
            @test_throws ArgumentError r2_constant_sigma(f_meta)
            msg_meta = try
                r2_constant_sigma(f_meta)
            catch e
                sprint(showerror, e)
            end
            @test occursin("meta_V", msg_meta)
        end

        @testset "NaN responses are dropped before computing R² (#1017)" begin
            yna = Vector{Float64}(randn(n))
            yna[1:5] .= NaN
            fna = drm(bf(@formula(yna ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; yna, x))
            r = r2_constant_sigma(fna)
            y = fna.obs[:mu]
            ŷ = fna.means[:mu]
            keep = isfinite.(y) .& isfinite.(ŷ)
            yk = y[keep]
            ŷk = ŷ[keep]
            ss_res = sum(abs2, yk .- ŷk)
            ss_tot = sum(abs2, yk .- (sum(yk) / length(yk)))
            @test r ≈ 1 - ss_res / ss_tot atol = 1e-10
        end
    end

    @testset "visualization.jl — corpairs and profile argument guards" begin
        Random.seed!(202611271)
        n = 200
        x = randn(n)
        z1 = randn(n)
        z2 = randn(n)
        ρt = 0.35
        e1 = 0.5 .* z1
        e2 = 0.6 .* (ρt .* z1 .+ sqrt(1 - ρt^2) .* z2)
        y = 0.2 .+ 0.4 .* x .+ e1
        y2 = -0.1 .+ 0.15 .* x .+ e2
        fitv = drm(
            bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
               rho12 = @formula(rho12 ~ x)),
            Gaussian();
            data = (; y, y2, x),
        )
        cd = corpairs_data(fitv)
        @test !cd.constant
        @test length(cd.rho) == n
        @test maximum(abs, cd.rho .- cd.rho[1]) > 0.01

        fitg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test_throws ArgumentError profile_curve(fitg, 2; npoints = 2)
        @test_throws ArgumentError parameter_surface(fitg, 1, 2; npoints = 1)
    end

    @testset "plotting_ext.jl — drmTMB-named aliases call the Makie stub" begin
        Random.seed!(202611272)
        n = 120
        x = randn(n)
        y = 0.5 .* x .+ randn(n)
        fitp = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test_throws MethodError plot_profile(fitp, 2)
        @test_throws MethodError plot_parameter_surface(fitp, 1, 2)
        bfit = drm(
            bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
               rho12 = @formula(rho12 ~ 1)),
            Gaussian();
            data = (; y, y2 = y .+ 0.1 .* randn(n), x),
        )
        @test_throws MethodError plot_corpairs(bfit)
    end

    @testset "rescale_response.jl — plan declines outside the simple slice" begin
        d = (; y = randn(40), x = randn(40), g = repeat(1:10, 4))
        f_uni = bf(@formula(y ~ x), @formula(sigma ~ 1))
        dk = (; y = d.y .* 1e6, x = d.x, g = d.g)
        @test DRModels._y_rescale_plan(
            f_uni, Gaussian(), (; data = dk, marginal = :VA), (:data, :g_tol),
        ) === nothing
        @test DRModels._y_rescale_plan(f_uni, Gaussian(), (; data = dk, method = :REML), (:data, :g_tol)) === nothing
        @test DRModels._y_rescale_plan(
            f_uni, Gaussian(), (; data = (y = [1.0, 2.0], x = [0.0, 1.0])), (:data, :g_tol),
        ) === nothing
    end
end