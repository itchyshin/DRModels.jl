using DRModels, Test, Random, LinearAlgebra
import Distributions
using Distributions: Normal, cdf

@testset "Binomial spatial SD regression" begin
    G, n_each = 48, 32
    group_index = repeat(1:G; inner = n_each)
    group = ["s$j" for j in group_index]
    coords = hcat(collect(range(0.0, 1.0; length = G)),
                  sin.(collect(range(0.0, 2pi; length = G))))
    # The Binomial spatial routes keep the legacy mean-range kernel (drmTMB has no binomial
    # spatial counterpart); the default of the shared builder is now drmTMB's native kernel.
    C = DRModels._spatial_covariance_from_coords(:site, G, coords, nothing;
                                                 legacy_mean_range = true)
    z_group = vcat(fill(-1.0, G ÷ 2), fill(1.0, G ÷ 2))
    z = repeat(z_group; inner = n_each)
    alpha = [log(0.55), 0.4]
    formula = bf(
        @formula(y ~ x + spatial(1 | group)),
        @formula(sd(group, spatial) ~ z),
    )

    for (link, seed) in ((:logit, 20261027), (:probit, 20261029), (:cloglog, 20261030))
        @testset "$link link" begin
            rng = MersenneTwister(seed)
            shuffle!(rng, z_group)
            z = repeat(z_group; inner = n_each)
            x = randn(rng, length(group))
            u = cholesky(Symmetric(C)).L * randn(rng, G)
            b = exp.(alpha[1] .+ alpha[2] .* z_group) .* u
            eta = -0.2 .+ 0.25 .* x .+ b[group_index]
            p = link === :logit ? 1 ./ (1 .+ exp.(-eta)) :
                link === :probit ? cdf.(Ref(Normal()), eta) : 1 .- exp.(-exp.(eta))
            y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(p)))
            data = (; y, x, z, group)
            fit = drm(formula, DRModels.Binomial(link = link);
                data, coords, se = true, g_tol = 1e-5)
            @test fit.converged
            @test isfinite(fit.loglik)
            @test coef(fit, :sd)[2] > 0
            @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.45
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
    # The Binomial spatial routes keep the legacy mean-range kernel (drmTMB has no binomial
    # spatial counterpart); the default of the shared builder is now drmTMB's native kernel.
    C = DRModels._spatial_covariance_from_coords(:site, G, coords, nothing;
                                                 legacy_mean_range = true)
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
            rng = MersenneTwister(seed)
            shuffle!(rng, z_group)
            z = repeat(z_group; inner = n_each)
            tau = exp.(alpha[1] .+ alpha[2] .* z_group)
            x = randn(rng, length(group))
            b = tau .* (cholesky(Symmetric(C)).L * randn(rng, G))
            eta = -0.2 .+ 0.25 .* x .+ x .* b[group_index]
            p = link === :logit ? 1 ./ (1 .+ exp.(-eta)) :
                link === :probit ? cdf.(Ref(Normal()), eta) :
                1 .- exp.(-exp.(eta))
            y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(p)))
            data = (; y, x, z, group)
            fit = drm(formula, DRModels.Binomial(link = link);
                data, coords, se = false, g_tol = 1e-5)

            @test fit.converged
            @test isfinite(fit.loglik)
            @test coef(fit, :sd)[2] > 0
            @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.45
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
