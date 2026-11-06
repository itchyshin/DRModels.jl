using DRModels
using Test, Random, Statistics
using Distributions: SkewNormal as DSkewNormal, logpdf
using QuadGK: quadgk
using StableRNGs: StableRNG

function _skewnormal_sd_moment_to_internal(mu, sigma, alpha)
    delta = alpha / sqrt(1 + alpha^2)
    omega = sigma / sqrt(1 - 2 * delta^2 / pi)
    xi = mu - omega * delta * sqrt(2 / pi)
    return xi, omega
end

function _skewnormal_sd_fixture(alpha_sd = [-0.55, 0.35])
    Random.seed!(20261021)
    G, m = 48, 12
    g = repeat(1:G, inner = m)
    z_group = repeat(repeat([-1.0, 1.0], inner = G ÷ 2), inner = m)
    habitat_group = repeat(["low", "high"], outer = G ÷ 2, inner = m)
    x = randn(G * m)
    sd_group = exp.(alpha_sd[1] .+ alpha_sd[2] .* z_group[1:m:end])
    b = sd_group .* randn(G)
    mu = 0.8 .+ 0.25 .* x .+ b[g]
    sigma, nu = 0.85, 2.5
    y = similar(mu)
    for i in eachindex(y)
        xi, omega = _skewnormal_sd_moment_to_internal(mu[i], sigma, nu)
        y[i] = rand(DSkewNormal(xi, omega, nu))
    end
    return (; y, x, g, z = z_group, habitat = habitat_group), alpha_sd
end

function _skewnormal_sd_scalar_oracle(data, fit)
    mu = fit.means[:mu]
    sigma = fit.scales[:sigma]
    nu = fit.scales[:nu]
    gidx = Int.(fit.scales[:sd_group_index])
    tau = fit.scales[:sd_group]
    ll = 0.0
    for g in eachindex(tau)
        rows = findall(==(g), gidx)
        integrand(b) = begin
            value = -log(tau[g]) - 0.5 * log(2π) - 0.5 * (b / tau[g])^2
            for i in rows
                xi, omega = _skewnormal_sd_moment_to_internal(mu[i] + b, sigma[i], nu[i])
                value += logpdf(DSkewNormal(xi, omega, nu[i]), data.y[i])
            end
            exp(value)
        end
        marginal, _ = quadgk(integrand, -Inf, Inf; rtol = 1e-9)
        ll += log(marginal)
    end
    return ll
end

function _skewnormal_sd_slope_fixture(alpha_sd = [-0.55, 0.40]; seed = 20261022)
    rng = StableRNG(seed)
    G, m = 48, 12
    g = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = repeat(z_group, inner = m)
    x = repeat(range(-1.5, 1.5; length = m), outer = G)
    sd_group = exp.(alpha_sd[1] .+ alpha_sd[2] .* z_group)
    b = sd_group .* randn(rng, G)
    mu = 0.8 .+ 0.35 .* x .+ x .* b[g]
    sigma, nu = 0.85, 2.5
    y = similar(mu)
    for i in eachindex(y)
        xi, omega = _skewnormal_sd_moment_to_internal(mu[i], sigma, nu)
        y[i] = rand(rng, DSkewNormal(xi, omega, nu))
    end
    return (; y, x, g, z), alpha_sd
end

function _skewnormal_sd_slope_oracle(data, fit)
    eta_fixed = fit.means[:mu]
    sigma = fit.scales[:sigma]
    nu = fit.scales[:nu]
    gidx = Int.(fit.scales[:sd_group_index])
    tau = fit.scales[:sd_group]
    ll = 0.0
    for g in eachindex(tau)
        rows = findall(==(g), gidx)
        integrand(b) = begin
            value = -log(tau[g]) - 0.5 * log(2π) - 0.5 * (b / tau[g])^2
            for i in rows
                xi, omega = _skewnormal_sd_moment_to_internal(
                    eta_fixed[i] + data.x[i] * b, sigma[i], nu[i])
                value += logpdf(DSkewNormal(xi, omega, nu[i]), data.y[i])
            end
            exp(value)
        end
        marginal, _ = quadgk(integrand, -Inf, Inf; rtol = 1e-9)
        ll += log(marginal)
    end
    return ll
end

@testset "SkewNormal ordinary random-intercept SD regression" begin
    data, alpha = _skewnormal_sd_fixture()
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(nu ~ 1), @formula(sd(g) ~ z + habitat))
    fit = drm(formula, SkewNormal(); data, nAGQ = 9)

    @test is_converged(fit)
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.3
    full_sd = predict_parameters(fit, data)[:sd]
    @test full_sd ≈ fit.scales[:sd_group][Int.(fit.scales[:sd_group_index])]
    high = data.habitat .== "high"
    high_data = (; (key => getproperty(data, key)[high] for key in keys(data))...)
    @test predict_parameters(fit, high_data)[:sd] ≈ full_sd[high]
    pred_se = predict_parameters(fit, data; se = true)
    @test all(isfinite, pred_se[:sd].se)
    @test fit.scales[:aghq_nodes] == [9.0]
    @test fit.loglik ≈ _skewnormal_sd_scalar_oracle(data, fit) atol = 1e-5

    marginal_simulator = DRModels._marginal_simulator_build(fit, data)
    @test marginal_simulator !== nothing
    marginal_draw = marginal_simulator(MersenneTwister(31))
    @test size(marginal_draw) == (length(data.y),)
    @test all(isfinite, marginal_draw)
    @test std([mean(marginal_draw[(g - 1) * 12 + 1:g * 12]) for g in 1:48]) > 0.35
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 9
    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(32),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    negative_data, negative_alpha = _skewnormal_sd_fixture([-0.55, -0.35])
    negative_fit = drm(formula, SkewNormal(); data = negative_data, nAGQ = 9)
    @test is_converged(negative_fit)
    @test sign(coef(negative_fit, :sd)[2]) == sign(negative_alpha[2])
    @test abs(coef(negative_fit, :sd)[2] - negative_alpha[2]) < 0.3

    bad_sd = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                @formula(nu ~ 1), @formula(sd(g) ~ x))
    @test_throws ArgumentError drm(bad_sd, SkewNormal(); data)
end

@testset "SkewNormal independent-slope SD regression" begin
    data, alpha = _skewnormal_sd_slope_fixture()
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(nu ~ 1), @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, SkewNormal(); data, nAGQ = 9)
    @test is_converged(fit)
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.3
    @test fit.scales[:aghq_nodes] == [9.0]
    @test fit.loglik ≈ _skewnormal_sd_slope_oracle(data, fit) atol = 1e-5

    predicted = predict_parameters(fit, data)
    expected_sd = fit.scales[:sd_group][Int.(fit.scales[:sd_group_index])]
    @test predicted[:sd] ≈ expected_sd
    @test all(isfinite, predict_parameters(fit, data; se = true)[:sd].se)
    b = ranef(fit)[:g]
    @test length(b) == length(unique(data.g))
    @test all(isfinite, b)

    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    ysim = simulator(StableRNG(20261023))
    @test length(ysim) == length(data.y)
    @test all(isfinite, ysim)
    first_row, opposite_slope_row = 1, 12
    marginal_draws = [simulator(StableRNG(20261030 + i)) for i in 1:500]
    empirical_cov = cov([draw[first_row] for draw in marginal_draws],
                        [draw[opposite_slope_row] for draw in marginal_draws])
    expected_cov = data.x[first_row] * data.x[opposite_slope_row] *
                   fit.scales[:sd_group][1]^2
    @test expected_cov < 0
    @test empirical_cov ≈ expected_cov atol = 0.20
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 9
    boot = bootstrap_result(fit; data, B = 1, rng = StableRNG(20261024),
                            failures = :error)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0

    negative_data, negative_alpha = _skewnormal_sd_slope_fixture(
        [-0.55, -0.40]; seed = 20261025)
    negative_fit = drm(formula, SkewNormal(); data = negative_data, nAGQ = 9)
    @test is_converged(negative_fit)
    @test sign(coef(negative_fit, :sd)[2]) == sign(negative_alpha[2])
    @test abs(coef(negative_fit, :sd)[2] - negative_alpha[2]) < 0.3

    wrong_target = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                      @formula(nu ~ 1), @formula(sd(g, coef(intercept)) ~ z))
    @test_throws ArgumentError drm(wrong_target, SkewNormal(); data)
    zero_slope = merge(data, (; x = zeros(length(data.x))))
    zero_formula = bf(@formula(y ~ 1 + (0 + x | g)), @formula(sigma ~ 1),
                      @formula(nu ~ 1), @formula(sd(g, coef(x)) ~ z))
    @test_throws ArgumentError drm(zero_formula, SkewNormal(); data = zero_slope)
end
