using DRModels, Test, Random, LinearAlgebra
import Distributions
using Distributions: Normal
import Serialization

@testset "Binomial related-matrix random-slope SD regression" begin
    rng = MersenneTwister(20261028)
    G, n_each = 64, 32
    group_index = repeat(1:G; inner = n_each)
    group = ["g$(i)" for i in group_index]
    z_group = vcat(fill(-1.0, G ÷ 2), fill(1.0, G ÷ 2))[randperm(rng, G)]
    z = repeat(z_group; inner = n_each)
    x = repeat(collect(range(-1.5, 1.5; length = n_each)), outer = G)

    marginal_scale = collect(range(0.8, 1.6; length = G))
    correlation = [0.4^abs(i - j) for i in 1:G, j in 1:G]
    K = Diagonal(sqrt.(marginal_scale)) * correlation *
        Diagonal(sqrt.(marginal_scale))
    unit_field = cholesky(Symmetric(K)).L * randn(rng, G)
    alpha = [-0.75, 0.5]
    slope_sd = exp.(alpha[1] .+ alpha[2] .* z_group)
    eta = -0.1 .+ 0.35 .* x .+ x .* (slope_sd .* unit_field)[group_index]
    probability = Distributions.logistic.(eta)
    successes = rand.(Ref(rng), Distributions.Bernoulli.(probability))
    failures = 1 .- successes
    data = (; successes, failures, x, z, group)
    formula = bf(
        @formula(cbind(successes, failures) ~ x + relmat(0 + x | group)),
        @formula(sd(group, relmat) ~ z),
    )

    fit = drm(formula, DRModels.Binomial(); data, K, se = true, g_tol = 1e-5)

    @test fit.converged
    @test isfinite(fit.loglik)
    @test all(isfinite, vcov(fit))
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.4
    sd_group = DRModels._binomial_sd_scale(hcat(ones(G), z_group) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ sd_group
    @test predict_parameters(fit, data)[:sd] ≈ sd_group[group_index]
    serialized = IOBuffer()
    Serialization.serialize(serialized, fit)
    seekstart(serialized)
    restored = Serialization.deserialize(serialized)
    @test predict_parameters(restored, data)[:sd] ≈ sd_group[group_index]

    draw_seed = 202610281
    simulator = DRModels._marginal_simulator_build(fit, data; K)
    actual_draw = simulator(MersenneTwister(draw_seed))
    oracle_rng = MersenneTwister(draw_seed)
    field = sd_group .* (cholesky(Symmetric(K)).L * randn(oracle_rng, G))
    eta_fixed = predict(fit, data; type = :link)
    expected_probability = Distributions.logistic.(
        eta_fixed .+ x .* field[group_index]
    )
    expected_draw = Float64[rand(oracle_rng, Distributions.Binomial(
        1, clamp(expected_probability[i], eps(), 1 - eps()))) for i in eachindex(x)]
    @test actual_draw == expected_draw

    boot = bootstrap_result(fit; data, B = 1, K,
        rng = MersenneTwister(202610282), failures = :error,
        check_converged = true)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end

@testset "Binomial probit and cloglog related-matrix random-slope SD regression" begin
    for (link, seed_offset) in ((:probit, 0), (:cloglog, 1))
        rng = MersenneTwister(20261050 + seed_offset)
        G, n_each = 64, 32
        matrix_order = vcat(collect(7:G), 1:6)
        group_index = repeat(1:G; inner = n_each)
        group = ["g$(matrix_order[j])" for j in group_index]
        z_group = vcat(fill(-1.0, G ÷ 2), fill(1.0, G ÷ 2))[randperm(rng, G)]
        z = repeat(z_group; inner = n_each)
        x = repeat(collect(range(-1.5, 1.5; length = n_each)), outer = G)

        kernel_scale = collect(range(0.8, 1.6; length = G))
        correlation = [0.4^abs(i - j) for i in 1:G, j in 1:G]
        K = Diagonal(sqrt.(kernel_scale)) * correlation *
            Diagonal(sqrt.(kernel_scale))
        unit_field = cholesky(Symmetric(K)).L * randn(rng, G)
        alpha = [-0.75, 0.5]
        slope_sd = exp.(alpha[1] .+ alpha[2] .* z_group)
        eta = -0.1 .+ 0.35 .* x .+ x .* (slope_sd .* unit_field)[group_index]
        probability = link === :probit ? Distributions.cdf.(Normal(), eta) :
                      -expm1.(-exp.(eta))
        successes = rand.(Ref(rng), Distributions.Bernoulli.(probability))
        failures = 1 .- successes
        data = (; successes, failures, x, z, group)
        formula = bf(
            @formula(cbind(successes, failures) ~ x + relmat(0 + x | group)),
            @formula(sd(group, relmat) ~ z),
        )

        fit = drm(formula, DRModels.Binomial(link = link);
                  data, K, se = false, g_tol = 1e-5)
        @test fit.converged
        @test isfinite(fit.loglik)
        @test coef(fit, :sd)[2] > 0
        @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.4
        group_sd = DRModels._binomial_sd_scale(
            hcat(ones(G), z_group) * coef(fit, :sd))
        @test predict_parameters(fit, data)[:sd] ≈ group_sd[group_index]

        draw_seed = 202610510 + seed_offset
        simulator = DRModels._marginal_simulator_build(fit, data; K)
        actual_draw = simulator(MersenneTwister(draw_seed))
        oracle_rng = MersenneTwister(draw_seed)
        field = group_sd .* (cholesky(Symmetric(K)).L * randn(oracle_rng, G))
        eta_fixed = predict(fit, data; type = :link)
        eta_draw = eta_fixed .+ x .* field[group_index]
        expected_probability = link === :probit ?
            Distributions.cdf.(Normal(), eta_draw) : -expm1.(-exp.(eta_draw))
        expected_draw = Float64[rand(oracle_rng, Distributions.Binomial(
            1, clamp(expected_probability[i], eps(), 1 - eps())))
            for i in eachindex(x)]
        @test actual_draw == expected_draw
    end
end
