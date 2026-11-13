using DRModels, Test, Random, LinearAlgebra
import Distributions

function _binomial_nonlogit_animal_sd_fixture(link; G = 128, m = 32,
                                               seed = link === :probit ? 20261027 : 20261028)
    rng = MersenneTwister(seed)
    order = randperm(rng, G)
    labels = ["animal_$(i)" for i in order]
    correlation = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    marginal_sd = collect(range(0.7, 1.5; length = G))
    A = Diagonal(marginal_sd) * correlation * Diagonal(marginal_sd)
    group = repeat(labels, inner = m)
    group_index = repeat(1:G, inner = m)
    z_group = collect(range(-1.0, 1.0; length = G))
    z = z_group[group_index]
    x = randn(rng, G * m)
    alpha = [-0.9, 0.55]
    latent = cholesky(Symmetric(A)).L * randn(rng, G)
    eta = -0.25 .+ 0.2 .* x .+
          exp.(alpha[1] .+ alpha[2] .* z_group)[group_index] .* latent[group_index]
    probability = if link === :probit
        Distributions.cdf.(Ref(Distributions.Normal()), eta)
    else
        -expm1.(-exp.(eta))
    end
    y = Float64[rand(rng, Distributions.Bernoulli(clamp(p, eps(), 1 - eps())))
                for p in probability]
    return (; data = (; y, x, z, g = group), A, alpha, z_group, m)
end

@testset "Binomial probit and cloglog animal SD regression" begin
    formula = bf(@formula(y ~ x + animal(1 | g)),
                 @formula(sd(g, animal) ~ z))
    for link in (:probit, :cloglog)
        @testset "$link link" begin
            fixture = _binomial_nonlogit_animal_sd_fixture(link)
            fit = drm(formula, Binomial(link = link);
                      data = fixture.data, A = fixture.A, se = false)
            Ainv = inv(Symmetric(fixture.A))
            precision_fit = drm(formula, Binomial(link = link);
                                data = fixture.data, Ainv, se = false)
            relmat_formula = bf(@formula(y ~ x + relmat(1 | g)),
                                @formula(sd(g, relmat) ~ z))
            relmat_fit = drm(relmat_formula, Binomial(link = link);
                             data = fixture.data, K = fixture.A, se = false)

            @test fit.converged
            @test precision_fit.converged
            @test relmat_fit.converged
            @test coef(fit, :sd)[2] > 0
            @test abs(coef(fit, :sd)[2] - fixture.alpha[2]) < 0.4
            @test coef(precision_fit, :sd) ≈ coef(fit, :sd) atol = 5e-3
            @test coef(relmat_fit, :sd) ≈ coef(fit, :sd) atol = 5e-3
            @test precision_fit.loglik ≈ fit.loglik atol = 2e-5
            @test relmat_fit.loglik ≈ fit.loglik atol = 2e-5
            predicted_sd = predict_parameters(fit, fixture.data)[:sd]
            expected_sd = exp.(hcat(ones(length(fixture.z_group)), fixture.z_group) *
                               coef(fit, :sd))
            @test length(predicted_sd) == length(fixture.data.y)
            @test all(isfinite, predicted_sd)
            @test predicted_sd ≈ expected_sd[repeat(1:length(expected_sd), inner = fixture.m)]
            eta_link = predict(fit, fixture.data; type = :link)
            expected_mean = link === :probit ?
                Distributions.cdf.(Ref(Distributions.Normal()), eta_link) :
                -expm1.(-exp.(eta_link))
            @test predict(fit, fixture.data) ≈ expected_mean
            @test fitted(fit; re = :zero) ≈ expected_mean
            @test marginal_parameters(fit)[:mu] ≈ expected_mean
            @test residuals(fit; re = :zero) ≈ fixture.data.y .- expected_mean
            sim_seed = 20261029
            observed_simulation = simulate(fit; re = :zero, rng = MersenneTwister(sim_seed))
            expected_rng = MersenneTwister(sim_seed)
            expected_simulation = Float64[rand(expected_rng,
                Distributions.Binomial(1, clamp(expected_mean[i], eps(), 1 - eps())))
                for i in eachindex(expected_mean)]
            @test observed_simulation == expected_simulation
        end
    end
end
