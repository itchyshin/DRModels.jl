using DRModels, Test, Random, LinearAlgebra, Serialization
import Distributions

function _binomial_nonlogit_animal_slope_fixture(link; G = 128, m = 32,
                                                  seed = link === :probit ? 20261030 : 20261031)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    labels = ["animal_$(i)" for i in order]
    group = repeat(labels, inner = m)
    group_index = repeat(1:G, inner = m)

    z_group = vcat(fill(-1.0, G ÷ 2), fill(1.0, G ÷ 2))[randperm(rng, G)]
    z = z_group[group_index]
    x_within = collect(range(-1.2, 1.2; length = m))
    x = repeat(x_within, outer = G)

    # Unit-diagonal AR(1) covariance keeps the supplied relatedness scale
    # separate from the predictor-dependent animal SD.
    A = [0.4^abs(i - j) for i in 1:G, j in 1:G]
    alpha = [-0.45, 0.55]
    latent = cholesky(Symmetric(A)).L * randn(rng, G)
    group_sd = exp.(alpha[1] .+ alpha[2] .* z_group)
    eta = -0.15 .+ 0.25 .* x .+ x .* (group_sd .* latent)[group_index]
    probability = if link === :probit
        Distributions.cdf.(Ref(Distributions.Normal()), eta)
    else
        -expm1.(-exp.(eta))
    end
    y = Float64[rand(rng, Distributions.Bernoulli(clamp(p, eps(), 1 - eps())))
                for p in probability]
    return (; data = (; y, x, z, g = group), A, alpha)
end

@testset "Binomial probit and cloglog animal-slope SD recovery" begin
    formula = bf(@formula(y ~ x + animal(0 + x | g)),
                 @formula(sd(g, animal) ~ z))

    for link in (:probit, :cloglog)
        @testset "$link link" begin
            fixture = _binomial_nonlogit_animal_slope_fixture(link)
            fit = drm(formula, Binomial(link = link);
                      data = fixture.data, A = fixture.A, se = false)
            Ainv = inv(Symmetric(fixture.A))
            precision_fit = drm(formula, Binomial(link = link);
                                data = fixture.data, Ainv, se = false)

            @test fit.converged
            @test precision_fit.converged
            @test abs(coef(fit, :sd)[1] - fixture.alpha[1]) < 0.35
            @test coef(fit, :sd)[2] > 0
            @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.35
            @test coef(precision_fit, :sd) ≈ coef(fit, :sd) atol = 5e-3
            @test precision_fit.loglik ≈ fit.loglik atol = 2e-5

            predicted_sd = predict_parameters(fit, fixture.data)[:sd]
            expected_sd = exp.(coef(fit, :sd)[1] .+
                                coef(fit, :sd)[2] .* fixture.data.z)
            @test length(predicted_sd) == length(fixture.data.y)
            @test all(isfinite, predicted_sd)
            @test predicted_sd ≈ expected_sd
            @test predict_parameters(precision_fit, fixture.data)[:sd] ≈
                  predicted_sd atol = 5e-3

            eta = predict(fit, fixture.data; type = :link)
            expected_mean = link === :probit ?
                Distributions.cdf.(Ref(Distributions.Normal()), eta) :
                -expm1.(-exp.(eta))
            @test predict(fit, fixture.data) ≈ expected_mean
            @test fitted(fit; re = :zero) ≈ expected_mean

            sim_seed = 20261032 + (link === :cloglog)
            observed_simulation = simulate(fit; re = :zero, rng = MersenneTwister(sim_seed))
            expected_rng = MersenneTwister(sim_seed)
            expected_simulation = Float64[rand(expected_rng,
                Distributions.Binomial(1, clamp(expected_mean[i], eps(), 1 - eps())))
                for i in eachindex(expected_mean)]
            @test observed_simulation == expected_simulation

            # A precision-input fit must retain the animal slope SD target
            # when the bootstrap draws a fresh related animal field and refits.
            precision_boot = bootstrap_result(precision_fit;
                data = fixture.data, Ainv, B = 1,
                rng = MersenneTwister(sim_seed + 100), failures = :skip)
            @test precision_boot.attempted == 1
            @test precision_boot.used == 1
            @test precision_boot.failed == 0

            buffer = IOBuffer()
            serialize(buffer, fit)
            seekstart(buffer)
            reloaded = deserialize(buffer)
            @test coef(reloaded, :sd) ≈ coef(fit, :sd) atol = 1e-12
            @test predict_parameters(reloaded, fixture.data)[:sd] ≈ predicted_sd
            @test predict(reloaded, fixture.data) ≈ expected_mean
        end
    end
end
