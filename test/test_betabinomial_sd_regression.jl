# Predictor-dependent SD for an ordinary BetaBinomial mean random intercept.
using DRModels, Test, Random, LinearAlgebra, QuadGK
import Distributions

function _betabinomial_sd_fixture(alpha; G = 48, m = 20, trials = 20,
                                  seed = 20261020)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    eta = 0.2 .+ 0.35 .* x .+ b[g]
    mu = 1 ./ (1 .+ exp.(-eta))
    sigma = 0.36
    precision = inv(sigma^2)
    successes = [rand(rng, Distributions.BetaBinomial(
        trials, mu[i] * precision, (1 - mu[i]) * precision))
        for i in eachindex(mu)]
    failures = trials .- successes
    @assert all(isfinite, zgroup)
    @assert rank(hcat(ones(G), zgroup)) == 2
    return (; successes = Float64.(successes), failures = Float64.(failures),
            x, z, zgroup, g), alpha
end

function _betabinomial_sd_slope_fixture(alpha; G = 60, m = 30, trials = 20,
                                        seed = 20261021)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    eta = 0.2 .+ 0.35 .* x .+ b[g] .* x
    mu = 1 ./ (1 .+ exp.(-eta))
    sigma = 0.36
    precision = inv(sigma^2)
    successes = [rand(rng, Distributions.BetaBinomial(
        trials, mu[i] * precision, (1 - mu[i]) * precision))
        for i in eachindex(mu)]
    failures = trials .- successes
    @assert rank(hcat(ones(G), zgroup)) == 2
    return (; successes = Float64.(successes), failures = Float64.(failures),
            x, z, zgroup, g), alpha
end

@testset "BetaBinomial predictor-dependent SD: ordinary random intercept" begin
    data, alpha = _betabinomial_sd_fixture([-1.15, 0.6])
    formula = bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                 @formula(sigma ~ 1), @formula(sd(g) ~ z))
    fit = drm(formula, BetaBinomial(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sigma(fit) ≈ fit.scales[:sigma]
    @test length(sigma(fit)) == length(data.g)
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.65
    Zg = hcat(ones(length(data.zgroup)), data.zgroup)
    group_sd = exp.(Zg * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ group_sd
    @test Int.(fit.scales[:sd_group_index]) == data.g
    @test predict_parameters(fit, data)[:sd] ≈ group_sd[data.g]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈
          log.(group_sd[data.g])
    explicit_intercept_formula = bf(
        @formula(cbind(successes, failures) ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g, coef(intercept)) ~ z))
    explicit_intercept_fit = drm(explicit_intercept_formula, BetaBinomial(); data, se = false)
    @test coef(explicit_intercept_fit, :sd) ≈ coef(fit, :sd) atol = 1e-4
    sd_prediction = predict_parameters(fit, data; se = true)[:sd]
    @test all(isfinite, sd_prediction.value)
    @test all(isfinite, sd_prediction.se)

    # Newdata can omit a training factor level. Prediction must reuse the
    # fitted contrast schema instead of rebuilding it from this subset.
    habitat = ifelse.(data.z .> 0, "open", "closed")
    factor_data = merge(data, (; habitat))
    factor_formula = bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                       @formula(sigma ~ 1), @formula(sd(g) ~ habitat))
    factor_fit = drm(factor_formula, BetaBinomial(); data = factor_data)
    one_level_newdata = (; g = [1, 1], x = [0.0, 0.5], habitat = ["closed", "closed"])
    factor_prediction = predict_parameters(factor_fit, one_level_newdata)[:sd]
    @test length(factor_prediction) == 2
    @test all(isfinite, factor_prediction)
    @test factor_prediction ≈ fill(exp(only(coef(factor_fit, :sd)[1:1])), 2)

    # Independent standard-normal integration at the fitted parameter vector.
    eta_mu = hcat(ones(length(data.x)), data.x) * coef(fit, :mu)
    phi_hat = exp(-2 * only(coef(fit, :sigma)))
    oracle_nll = 0.0
    for gi in eachindex(group_sd)
        rows = findall(==(gi), data.g)
        integrand(u) = begin
            ll = sum(i -> begin
                mu_i = 1 / (1 + exp(-(eta_mu[i] + group_sd[gi] * u)))
                Distributions.logpdf(Distributions.BetaBinomial(
                    20, mu_i * phi_hat, (1 - mu_i) * phi_hat),
                    data.successes[i])
            end, rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-8)
        oracle_nll -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle_nll atol = 0.08

    # The intercept-only SD route must match the existing constant-SD route.
    constant = bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                  @formula(sigma ~ 1), @formula(sd(g) ~ 1))
    # Same node count as the plain `(1 | g)` route, so only the likelihood assembly is compared.
    fit_sd_constant = DRModels._drm_betabinomial_sd_regression(constant, BetaBinomial(); data,
                                                           se = false,
                                                           nAGQ = DRModels._RANEF1D_AGHQ_K)
    fit_legacy = drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                        @formula(sigma ~ 1)), BetaBinomial(); data, se = false)
    @test fit_sd_constant.loglik ≈ fit_legacy.loglik atol = 1e-5
    @test coef(fit_sd_constant, :mu) ≈ coef(fit_legacy, :mu) atol = 1e-4
    @test coef(fit_sd_constant, :sigma) ≈ coef(fit_legacy, :sigma) atol = 1e-4
    @test only(coef(fit_sd_constant, :sd)) ≈
          only(coef(fit_legacy, :resd)) atol = 1e-4

    varying_sd_predictor = copy(data.z)
    varying_sd_predictor[1] += 0.25
    @test_throws ArgumentError drm(formula, BetaBinomial();
        data = merge(data, (; z = varying_sd_predictor)), se = false)
    wrong_member_formula = bf(@formula(cbind(successes, failures) ~ x + (1 | g)),
                              @formula(sigma ~ 1),
                              @formula(sd(g, coef(x)) ~ z))
    @test_throws ArgumentError drm(wrong_member_formula, BetaBinomial(); data, se = false)
    zero_trial_data = merge(data, (; successes = zeros(length(data.successes)),
                                    failures = zeros(length(data.failures))))
    @test_throws ArgumentError drm(formula, BetaBinomial(); data = zero_trial_data, se = false)
end

@testset "BetaBinomial predictor-dependent SD: independent random slope" begin
    data, alpha = _betabinomial_sd_slope_fixture([-1.35, 0.6])
    formula = bf(@formula(cbind(successes, failures) ~ x + (0 + x | g)),
                 @formula(sigma ~ 1), @formula(sd(g) ~ z))
    fit = drm(formula, BetaBinomial(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.7
    group_sd = exp.(hcat(ones(length(data.zgroup)), data.zgroup) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ group_sd
    @test predict_parameters(fit, data)[:sd] ≈ group_sd[data.g]

    # Independent scalar integration checks that each random coefficient is
    # multiplied by its row's slope loading inside the group likelihood.
    eta_mu = hcat(ones(length(data.x)), data.x) * coef(fit, :mu)
    phi_hat = exp(-2 * only(coef(fit, :sigma)))
    oracle_nll = 0.0
    for gi in eachindex(group_sd)
        rows = findall(==(gi), data.g)
        integrand(u) = begin
            ll = sum(i -> begin
                mu_i = 1 / (1 + exp(-(eta_mu[i] + data.x[i] * group_sd[gi] * u)))
                Distributions.logpdf(Distributions.BetaBinomial(
                    20, mu_i * phi_hat, (1 - mu_i) * phi_hat),
                    data.successes[i])
            end, rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-8)
        oracle_nll -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle_nll atol = 0.12
end
