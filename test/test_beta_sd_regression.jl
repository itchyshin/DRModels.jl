# Predictor-dependent SD for ordinary Beta location intercepts and numeric slopes.
using DRModels, Random, Test, QuadGK
import Distributions

function _beta_sd_fixture(alpha; G = 40, m = 18, seed = 202610181)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = 1 ./ (1 .+ exp.(-(0.25 .+ 0.2 .* x .+ b[g])))
    sigma = 0.32
    phi = 1 / sigma^2
    y = [rand(rng, Distributions.Beta(mu[i] * phi, (1 - mu[i]) * phi))
         for i in eachindex(mu)]
    return (; y = Float64.(y), x, z, g, zgroup), alpha
end

function _beta_sd_slope_fixture(alpha; G = 72, m = 28, seed = 202610187)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = repeat(range(-1.25, 1.25; length = m), outer = G)
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = 1 ./ (1 .+ exp.(-(0.2 .+ 0.25 .* x .+ b[g] .* x)))
    sigma = 0.3
    phi = 1 / sigma^2
    y = [rand(rng, Distributions.Beta(mu[i] * phi, (1 - mu[i]) * phi))
         for i in eachindex(mu)]
    return (; y = Float64.(y), x, z, g, zgroup), alpha
end

@testset "Beta direct-SD ordinary intercept recovery and public outputs" begin
    data, alpha = _beta_sd_fixture([-0.55, 0.35])
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, coef(intercept)) ~ z))
    fit = drm(formula, Beta(); data)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.7
    Zg = hcat(ones(40), data.zgroup)
    @test fit.scales[:sd_group] ≈ exp.(Zg * coef(fit, :sd))
    @test Int.(fit.scales[:sd_group_index]) == data.g
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈ log.(fit.scales[:sd])
    pred_sd = predict_parameters(fit, data; se = true)[:sd]
    @test all(isfinite, pred_sd.value)
    @test all(isfinite, pred_sd.se)

    # Independent standard-normal integration at the fitted parameter vector.
    beta_mu = coef(fit, :mu); log_sigma = only(coef(fit, :sigma))
    alpha_hat = coef(fit, :sd)
    eta_mu = hcat(ones(length(data.y)), data.x) * beta_mu
    group_sd = exp.(Zg * alpha_hat)
    oracle = 0.0
    for gi in eachindex(group_sd)
        rows = findall(==(gi), data.g)
        integrand(u) = begin
            b = group_sd[gi] * u
            ll = sum(i -> Distributions.logpdf(Distributions.Beta(
                (1 / (1 + exp(-(eta_mu[i] + b)))) * exp(-2log_sigma),
                (1 - 1 / (1 + exp(-(eta_mu[i] + b)))) * exp(-2log_sigma)),
                data.y[i]), rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
        oracle -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle atol = 0.08

    constant = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                  @formula(sd(g) ~ 1))
    fit_sd_constant = drm(constant, Beta(); data)
    fit_legacy = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)),
                     Beta(); data)
    @test fit_sd_constant.loglik ≈ fit_legacy.loglik atol = 1e-5
    @test coef(fit_sd_constant, :mu) ≈ coef(fit_legacy, :mu) atol = 1e-4
    @test coef(fit_sd_constant, :sigma) ≈ coef(fit_legacy, :sigma) atol = 1e-4
    @test only(coef(fit_sd_constant, :sd)) ≈ only(coef(fit_legacy, :resd)) atol = 1e-4

    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(202610186), failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    # Marginal simulation must draw from the fitted Beta model after drawing a
    # fresh random intercept, rather than returning only its conditional mean.
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    rng_sim = MersenneTwister(202610189)
    draw = simulator(rng_sim)
    rng_oracle = MersenneTwister(202610189)
    gidx = Int.(fit.scales[:sd_group_index])
    sd_group = fit.scales[:sd_group]
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    u = sd_group .* randn(rng_oracle, length(sd_group))
    mu_star = 1 ./ (1 .+ exp.(-clamp.(eta_fixed .+ u[gidx], -30.0, 30.0)))
    expected = DRModels._simulate_once(fit, rng_oracle; mu = mu_star)
    @test draw == expected
end

@testset "Beta direct-SD independent slope recovery and likelihood" begin
    data, alpha = _beta_sd_slope_fixture([-0.65, 0.4])
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, Beta(); data)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test last(fit.coefnames)[2] == ["(Intercept)", "z"]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    Zg = hcat(ones(length(data.zgroup)), data.zgroup)
    @test fit.scales[:sd_group] ≈ exp.(Zg * coef(fit, :sd))
    @test Int.(fit.scales[:sd_group_index]) == data.g
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd_group][data.g]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈
          log.(fit.scales[:sd_group][data.g])

    # Independent standard-normal integration for the slope loading x.
    beta_mu = coef(fit, :mu); log_sigma = only(coef(fit, :sigma))
    alpha_hat = coef(fit, :sd)
    eta_mu = hcat(ones(length(data.y)), data.x) * beta_mu
    group_sd = exp.(Zg * alpha_hat)
    oracle = 0.0
    for gi in eachindex(group_sd)
        rows = findall(==(gi), data.g)
        integrand(u) = begin
            ll = sum(i -> begin
                eta = eta_mu[i] + group_sd[gi] * u * data.x[i]
                mu = 1 / (1 + exp(-eta))
                precision = exp(-2log_sigma)
                Distributions.logpdf(Distributions.Beta(
                    mu * precision, (1 - mu) * precision), data.y[i])
            end, rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
        oracle -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle atol = 0.08

    # A constant coefficient SD is recovered by the intercept-only model when
    # the same constant SD is applied to a unit loading.
    intercept_data, _ = _beta_sd_fixture([-0.65, 0.4]; G = 72, m = 28,
                                         seed = 202610187)
    intercept_data = merge(intercept_data, (; slope = ones(length(data.x))))
    intercept_formula = bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1),
                           @formula(sd(g, coef(intercept)) ~ 1))
    slope_formula = bf(@formula(y ~ 1 + (0 + slope | g)), @formula(sigma ~ 1),
                       @formula(sd(g, coef(slope)) ~ 1))
    intercept_fit = drm(intercept_formula, Beta(); data = intercept_data)
    slope_constant_fit = drm(slope_formula, Beta(); data = intercept_data)
    @test slope_constant_fit.loglik ≈ intercept_fit.loglik atol = 1e-5
    @test coef(slope_constant_fit, :sd) ≈ coef(intercept_fit, :sd) atol = 1e-4

    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(202610188),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end

@testset "Beta random-effect SD regression refuses unsupported designs" begin
    data, _ = _beta_sd_fixture([-0.55, 0.35]; G = 12, m = 6,
                               seed = 202610184)
    base = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
              @formula(sd(g) ~ z))
    varying = merge(data, (; z = data.z .+ randn(MersenneTwister(202610185),
                                                   length(data.z))))
    @test_throws ArgumentError drm(base, Beta(); data = varying)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ z), @formula(sd(g) ~ z)), Beta(); data)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 + x | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)), Beta(); data)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | g)),
        @formula(sigma ~ 1), @formula(sd(g, coef(other)) ~ z)), Beta(); data)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | g)),
        @formula(sigma ~ 1), @formula(sd(g, coef(x)) ~ z)), Beta();
        data = merge(data, (; x = string.(data.x))))
    missing_data = merge(data, (; y = Union{Missing,Float64}[missing; data.y[2:end]]))
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | g)),
        @formula(sigma ~ 1), @formula(sd(g, coef(x)) ~ z)), Beta();
        data = missing_data)
end
