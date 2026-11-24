using DRModels, Test, Random, LinearAlgebra
import Distributions
using Distributions: Normal, cdf
include(joinpath(@__DIR__, "fixtures", "binomial_spatial_sd", "gen_binomial_spatial_sd.jl"))

@testset "Binomial spatial SD regression" begin
    G, n_each = 48, 32
    group_index = repeat(1:G; inner = n_each)
    group = ["s$j" for j in group_index]
    coords = hcat(collect(range(0.0, 1.0; length = G)),
                  sin.(collect(range(0.0, 2pi; length = G))))
    # The Binomial spatial routes use drmTMB's native fixed kernel (median positive distance,
    # 1e-6 ridge), which is the default of the shared builder.
    C = DRModels._spatial_covariance_from_coords(:site, G, coords, nothing)
    Lchol = Matrix(cholesky(Symmetric(C)).L)
    z_group = vcat(fill(-1.0, G ÷ 2), fill(1.0, G ÷ 2))
    z = repeat(z_group; inner = n_each)
    alpha = [log(0.55), 0.4]
    formula = bf(
        @formula(y ~ x + spatial(1 | group)),
        @formula(sd(group, spatial) ~ z),
    )

    for (link, seed) in ((:logit, 20261027), (:probit, 20261029), (:cloglog, 20261030))
        @testset "$link link" begin
            # Version-independent draw (splitmix64): the Random stdlib's shuffle!/randn/rand
            # return different streams for one seed on Julia 1.10 and 1.12.
            d = gen_binomial_spatial_sd(link, seed; Lchol = Lchol, alpha = alpha, slope = false)
            (; y, x, z, group) = d
            z_group = d.z_group
            data = (; y, x, z, group)
            fit = drm(formula, DRModels.Binomial(link = link);
                data, coords, se = true, g_tol = 1e-5)
            @test fit.converged
            @test isfinite(fit.loglik)
            # Recovery of alpha[2] is a distribution-level claim: see the seed-median testset below.
            @test fit.scales[:sd_group] ≈
                DRModels._binomial_structured_sd_group_scale(
                    hcat(ones(G), z_group), coef(fit, :sd))
            @test predict_parameters(fit, data)[:sd] ≈
                fit.scales[:sd_group][group_index] atol = 1e-5

            if link === :logit
                extreme_z_group = repeat([-100.0, 100.0], inner = G ÷ 2)
                extreme_data = merge(data, (; z = extreme_z_group[group_index]))
                extreme_eta = hcat(ones(G), extreme_z_group) * coef(fit, :sd)
                expected_extreme_sd =
                    DRModels._binomial_structured_sd_group_scale(
                        hcat(ones(G), extreme_z_group), coef(fit, :sd))
                observed_extreme_sd =
                    predict_parameters(fit, extreme_data; se = false)[:sd]
                @test observed_extreme_sd ≈ expected_extreme_sd[group_index]
                @test maximum(observed_extreme_sd) < exp(maximum(extreme_eta))
            end

            # Seeded marginal simulation checks that spatial covariance and
            # predicted group scales are used by the public simulator.
            draw_seed = seed + 1
            simulator = DRModels._marginal_simulator_build(fit, data; coords)
            actual_draw = simulator(MersenneTwister(draw_seed))
            oracle_rng = MersenneTwister(draw_seed)
            group_sd = DRModels._binomial_structured_sd_group_scale(
                hcat(ones(G), z_group), coef(fit, :sd))
            latent = group_sd .* (cholesky(Symmetric(C)).L * randn(oracle_rng, G))
            eta_fixed = predict(fit, data; type = :link)
            expected_p = link === :logit ? 1 ./ (1 .+ exp.(-(eta_fixed .+ latent[group_index]))) :
                link === :probit ? cdf.(Ref(Normal()), eta_fixed .+ latent[group_index]) :
                1 .- exp.(-exp.(eta_fixed .+ latent[group_index]))
            expected_draw = Float64[rand(oracle_rng, Distributions.Binomial(
                1, clamp(expected_p[i], eps(), 1 - eps()))) for i in eachindex(y)]
            @test actual_draw == expected_draw
        end
    end
end

@testset "Binomial spatial random-slope SD regression" begin
    G, n_each = 48, 32
    group_index = repeat(1:G; inner = n_each)
    group = ["s$j" for j in group_index]
    axis = collect(range(0.0, 1.0; length = G))
    coords = hcat(axis, sin.(2pi .* axis))
    # The Binomial spatial routes use drmTMB's native fixed kernel (median positive distance,
    # 1e-6 ridge), which is the default of the shared builder.
    C = DRModels._spatial_covariance_from_coords(:site, G, coords, nothing)
    Lchol = Matrix(cholesky(Symmetric(C)).L)
    z_group = vcat(fill(-1.0, G ÷ 2), fill(1.0, G ÷ 2))
    alpha = [log(0.6), 0.4]
    formula = bf(
        @formula(y ~ x + spatial(0 + x | group)),
        @formula(sd(group, spatial) ~ z),
    )
    invalid_data = (; y = zeros(G * n_each),
        x = repeat(range(-1.0, 1.0; length = G); inner = n_each),
        z = repeat(z_group; inner = n_each), group)

    for (link, seed) in ((:logit, 2026102732), (:probit, 2026102733),
                         (:cloglog, 2026102734))
        @testset "$link slope" begin
            d = gen_binomial_spatial_sd(link, seed; Lchol = Lchol, alpha = alpha, slope = true)
            (; y, x, z, group) = d
            z_group = d.z_group
            data = (; y, x, z, group)
            fit = drm(formula, DRModels.Binomial(link = link);
                data, coords, se = false, g_tol = 1e-5)

            @test fit.converged
            @test isfinite(fit.loglik)
            # Recovery of alpha[2] is a distribution-level claim: see the seed-median testset below.
            @test fit.scales[:sd_group] ≈
                DRModels._binomial_structured_sd_group_scale(
                    hcat(ones(G), z_group), coef(fit, :sd))
            @test predict_parameters(fit, data)[:sd] ≈
                fit.scales[:sd_group][group_index] atol = 1e-5

            # Compare the public marginal simulator to an independently
            # generated field using x * b[group], not b[group] alone.
            draw_seed = seed + 1
            simulator = DRModels._marginal_simulator_build(fit, data; coords)
            actual = simulator(MersenneTwister(draw_seed))
            oracle_rng = MersenneTwister(draw_seed)
            fitted_tau = fit.scales[:sd_group]
            latent = fitted_tau .* (cholesky(Symmetric(C)).L * randn(oracle_rng, G))
            eta_fixed = predict(fit, data; type = :link)
            eta_draw = eta_fixed .+ x .* latent[group_index]
            p_draw = link === :logit ? 1 ./ (1 .+ exp.(-eta_draw)) :
                link === :probit ? cdf.(Ref(Normal()), eta_draw) :
                1 .- exp.(-exp.(eta_draw))
            expected = Float64[rand(oracle_rng, Distributions.Binomial(
                1, clamp(p_draw[i], eps(), 1 - eps()))) for i in eachindex(y)]
            @test actual == expected
        end
    end

    @test_throws ArgumentError drm(formula, DRModels.Binomial();
        data = invalid_data, coords, se = false)
end

@testset "Binomial structured SD scales use the fitted smooth clamp" begin
    Zg = [1.0 -20.0; 1.0 20.0]
    alpha = [0.0, 1.0]
    @test DRModels._binomial_structured_sd_group_scale(Zg, alpha) ≈
        exp.([-12.0 - 3tanh(8 / 3), 12.0 + 3tanh(8 / 3)])
end

# A single binary draw (48 sites x 32 rows, one spatial field) identifies the SD slope alpha[2]
# poorly: over 80 seeds per cell (3 links x intercept/slope designs, identical on Julia 1.10 and
# 1.12 with the splitmix64 generator) |alpha2_hat - 0.4| < 0.45 holds in only 79-95% of draws
# (some draws run to the boundary), so a per-seed bound is a seed lottery. The median over 15
# seeds is stable: its 5-95% range over resamples of 15 of those 80 fits was 0.35-0.59 in every
# cell, so |median - 0.4| < 0.45 has a margin of at least 2x the measured spread.
@testset "Binomial spatial SD regression: seed-median recovery of the SD slope" begin
    G, n_each = 48, 32
    axis = collect(range(0.0, 1.0; length = G))
    for (design, slope) in (("random intercept", false), ("random slope", true))
        coords = slope ? hcat(axis, sin.(2pi .* axis)) :
                         hcat(axis, sin.(collect(range(0.0, 2pi; length = G))))
        C = DRModels._spatial_covariance_from_coords(:site, G, coords, nothing)
        Lchol = Matrix(cholesky(Symmetric(C)).L)
        alpha = slope ? [log(0.6), 0.4] : [log(0.55), 0.4]
        formula = slope ?
            bf(@formula(y ~ x + spatial(0 + x | group)), @formula(sd(group, spatial) ~ z)) :
            bf(@formula(y ~ x + spatial(1 | group)), @formula(sd(group, spatial) ~ z))
        for link in (:logit, :probit, :cloglog)
            @testset "$design, $link" begin
                est = Float64[]
                for seed in 1:15
                    d = gen_binomial_spatial_sd(link, seed; Lchol, alpha, slope)
                    fit = drm(formula, DRModels.Binomial(link = link);
                        data = (; y = d.y, x = d.x, z = d.z, group = d.group),
                        coords, se = false, g_tol = 1e-5)
                    push!(est, coef(fit, :sd)[2])
                end
                @test all(isfinite, est)
                @test abs(sort(est)[8] - alpha[2]) < 0.45
            end
        end
    end
end
