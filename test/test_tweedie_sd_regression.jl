# Predictor-dependent SD regression for Tweedie mean random intercepts and slopes.
using DRModels, Test, Random, LinearAlgebra, QuadGK
import Distributions

function _r_tweedie_sd(rng, mu, sigma, power)
    phi = sigma^2
    lambda = mu^(2 - power) / (phi * (2 - power))
    count = rand(rng, Distributions.Poisson(lambda))
    count == 0 && return 0.0
    shape = count * (2 - power) / (power - 1)
    scale = phi * (power - 1) * mu^(power - 1)
    return rand(rng, Distributions.Gamma(shape, scale))
end

function _tweedie_sd_fixture(; alpha = [-0.8, 0.65], G = 40, m = 3,
                              seed = 20261020)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[group]
    x = rand(rng, length(group)) .* 2 .- 1
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    mu = exp.(0.4 .+ 0.25 .* x .+ b[group])
    sigma = 0.7
    power = 1.5
    y = [_r_tweedie_sd(rng, mu[i], sigma, power) for i in eachindex(mu)]
    return (; y, x, z, group, z_group), alpha
end

@testset "Tweedie predictor-dependent SD: ordinary random intercept" begin
    data, alpha = _tweedie_sd_fixture()
    formula = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                 @formula(nu ~ 1), @formula(sd(group) ~ z))
    fit = drm(formula, Tweedie(); data, g_tol = 1e-5)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :nu, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.75
    group_sd = exp.(hcat(ones(length(data.z_group)), data.z_group) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ group_sd
    @test fit.scales[:sd] ≈ group_sd[data.group]
    @test predict_parameters(fit, data)[:sd] ≈ group_sd[data.group]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈ log.(group_sd[data.group])

    varying = merge(data, (; z = collect(range(-1.0, 1.0; length = length(data.z)))))
    @test_throws ArgumentError drm(formula, Tweedie(); data = varying, g_tol = 1e-5)

    # A constant SD formula must reproduce the existing random-intercept route.
    constant = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                  @formula(nu ~ 1), @formula(sd(group) ~ 1))
    fit_constant = drm(constant, Tweedie(); data, g_tol = 1e-5)
    fit_legacy = drm(bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                        @formula(nu ~ 1)), Tweedie(); data, g_tol = 1e-5)
    @test loglik(fit_constant) ≈ loglik(fit_legacy) atol = 1e-5
    @test coef(fit_constant, :mu) ≈ coef(fit_legacy, :mu) atol = 1e-4
    @test coef(fit_constant, :sigma) ≈ coef(fit_legacy, :sigma) atol = 1e-4
    @test coef(fit_constant, :nu) ≈ coef(fit_legacy, :nu) atol = 1e-4
    @test only(coef(fit_constant, :sd)) ≈ only(coef(fit_legacy, :resd)) atol = 1e-4
end

function _tweedie_sd_slope_fixture(; alpha = [-0.75, 0.5], G = 40, m = 4,
                                    seed = 20261021)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[group]
    x = rand(rng, length(group)) .* 2 .- 1
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    mu = exp.(0.35 .+ 0.2 .* x .+ x .* b[group])
    sigma = 0.7
    power = 1.5
    y = [_r_tweedie_sd(rng, mu[i], sigma, power) for i in eachindex(mu)]
    return (; y, x, z, group, z_group), alpha
end

@testset "Tweedie predictor-dependent SD: independent mean slope" begin
    data, alpha = _tweedie_sd_slope_fixture()
    formula = bf(@formula(y ~ x + (0 + x | group)), @formula(sigma ~ 1),
                 @formula(nu ~ 1), @formula(sd(group, coef(x)) ~ z))
    fit = drm(formula, Tweedie(); data, g_tol = 1e-5)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :nu, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.75
    group_sd = exp.(hcat(ones(length(data.z_group)), data.z_group) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ group_sd
    @test fit.scales[:sd] ≈ group_sd[data.group]
    @test predict_parameters(fit, data)[:sd] ≈ group_sd[data.group]

    wrong_target = bf(@formula(y ~ x + (0 + x | group)), @formula(sigma ~ 1),
                      @formula(nu ~ 1), @formula(sd(group, coef(intercept)) ~ z))
    @test_throws ArgumentError drm(wrong_target, Tweedie(); data, g_tol = 1e-5)
    zero_loading = merge(data, (; x = zeros(length(data.x))))
    @test_throws ArgumentError drm(formula, Tweedie(); data = zero_loading, g_tol = 1e-5)

    # Constant SD regression must retain the pre-existing random-slope target.
    constant = bf(@formula(y ~ x + (0 + x | group)), @formula(sigma ~ 1),
                  @formula(nu ~ 1), @formula(sd(group, coef(x)) ~ 1))
    fit_constant = drm(constant, Tweedie(); data, g_tol = 1e-5)
    fit_legacy = drm(bf(@formula(y ~ x + (0 + x | group)), @formula(sigma ~ 1),
                        @formula(nu ~ 1)), Tweedie(); data, g_tol = 1e-5)
    @test loglik(fit_constant) ≈ loglik(fit_legacy) atol = 1e-5
    @test coef(fit_constant, :mu) ≈ coef(fit_legacy, :mu) atol = 1e-4
    @test coef(fit_constant, :sigma) ≈ coef(fit_legacy, :sigma) atol = 1e-4
    @test coef(fit_constant, :nu) ≈ coef(fit_legacy, :nu) atol = 1e-4
    @test only(coef(fit_constant, :sd)) ≈
          only(coef(fit_legacy, :resd)) atol = 1e-4
end
