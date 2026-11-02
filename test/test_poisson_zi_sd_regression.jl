using DRModels, Test, Random, QuadGK, LinearAlgebra
import Distributions

function _poisson_zi_sd_fixture(; G = 80, m = 16)
    rng = MersenneTwister(20261024)
    g = repeat(1:G, inner = m)
    z_group = repeat([0.0, 1.0], inner = G ÷ 2)
    z = z_group[g]
    x = randn(rng, G * m)
    alpha = [-0.50, 0.40]
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    zi_probability = Distributions.logistic.(-1.5 .* ones(G * m))
    lambda = exp.(0.35 .+ 0.25 .* x .+ b[g])
    y = Float64.([rand(rng) < zi_probability[i] ? 0 :
        rand(rng, Distributions.Poisson(lambda[i])) for i in eachindex(lambda)])
    return (; y, x, z, g), alpha
end

function _poisson_zi_sd_slope_fixture(; G = 80, m = 16)
    rng = MersenneTwister(20261025)
    g = repeat(1:G, inner = m)
    z_group = repeat([0.0, 1.0], inner = G ÷ 2)
    z = z_group[g]
    x = randn(rng, G * m)
    alpha = [-0.50, 0.40]
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    zi_probability = Distributions.logistic.(-1.5 .* ones(G * m))
    lambda = exp.(0.35 .+ 0.25 .* x .+ b[g] .* x)
    y = Float64.([rand(rng) < zi_probability[i] ? 0 :
        rand(rng, Distributions.Poisson(lambda[i])) for i in eachindex(lambda)])
    return (; y, x, z, g), alpha
end

@testset "ZIP independent mean-slope SD regression" begin
    data, alpha = _poisson_zi_sd_slope_fixture()
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(zi ~ 1),
                 @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, Poisson(); data, nAGQ = 9)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :zi, :sd]
    @test length(coef(fit, :sd)) == 2
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(80), [zeros(40); ones(40)]) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈ log.(fit.scales[:sd])
    zero_loading = merge(data, (x = zeros(length(data.x)),))
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (0 + x | g)), @formula(zi ~ 1),
           @formula(sd(g, coef(x)) ~ z)), Poisson(); data = zero_loading, nAGQ = 3)

    profile_ci = confint(fit; method = :profile, parm = :sd)
    @test length(profile_ci) == 2
    @test all(row -> isfinite(row.lower) && isfinite(row.upper), profile_ci)

    # Independent direct integration for the scalar slope effect in each group.
    theta = [0.35, 0.25, -1.5, -0.50, 0.40]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:80
        rows = findall(==(gi), data.g)
        sdg = exp(theta[4] + theta[5] * data.z[first(rows)])
        integrand = u -> begin
            ll = 0.0
            for i in rows
                eta = theta[1] + theta[2] * data.x[i] + sdg * u * data.x[i]
                lambda = exp(eta)
                log_pi = -log1p(exp(1.5))
                log_one_minus_pi = -log1p(exp(-1.5))
                ll += data.y[i] == 0 ?
                    log(exp(log_pi) + exp(log_one_minus_pi - lambda)) :
                    log_one_minus_pi + data.y[i] * eta - lambda -
                        Distributions.loggamma(data.y[i] + 1)
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-10)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-3)

    # Check that marginal simulation uses x as the random-coefficient loading.
    seed = 2026102519
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    simulated = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    latent = fit.scales[:sd_group] .* randn(oracle_rng, 80)
    eta_fixed = predict(fit, data; type = :link)
    expected = Float64[
        rand(oracle_rng) < fit.scales[:zi][i] ? 0 :
            rand(oracle_rng, Distributions.Poisson(
                exp(eta_fixed[i] + latent[data.g[i]] * data.x[i])))
        for i in eachindex(data.y)
    ]
    @test simulated == expected

    boot = bootstrap_result(fit; data, B = 1,
                            rng = MersenneTwister(2026102520), failures = :skip)
    @test boot.attempted == 1
    @test boot.used + boot.failed == 1
end

@testset "ZIP mean random-intercept SD regression" begin
    data, alpha = _poisson_zi_sd_fixture()
    formula = bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1),
                 @formula(sd(g) ~ z))
    fit = drm(formula, Poisson(); data, nAGQ = 9)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :zi, :sd]
    @test length(coef(fit, :sd)) == 2
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(80), [zeros(40); ones(40)]) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:zi] ≈ fit.scales[:zi]
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈ log.(fit.scales[:sd])
    profile_ci = confint(fit; method = :profile, parm = :sd)
    @test length(profile_ci) == 2
    @test all(row -> isfinite(row.lower) && isfinite(row.upper), profile_ci)

    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (0 + x | g)), @formula(zi ~ 1),
           @formula(sd(g) ~ z)), Poisson(); data, nAGQ = 3)

    # Independent direct one-dimensional integration of each ZIP group likelihood.
    theta = [0.35, 0.25, -1.5, -0.50, 0.40]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:80
        rows = findall(==(gi), data.g)
        sdg = exp(theta[4] + theta[5] * data.z[first(rows)])
        integrand = u -> begin
            ll = 0.0
            for i in rows
                eta = theta[1] + theta[2] * data.x[i] + sdg * u
                lambda = exp(eta)
                log_pi = -log1p(exp(1.5))
                log_one_minus_pi = -log1p(exp(-1.5))
                ll += data.y[i] == 0 ?
                    log(exp(log_pi) + exp(log_one_minus_pi - lambda)) :
                    log_one_minus_pi + data.y[i] * eta - lambda -
                        Distributions.loggamma(data.y[i] + 1)
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-10)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-3)

    # The generic marginal simulator must redraw both group effects and ZIP rows.
    seed = 2026102419
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    simulated = simulator(MersenneTwister(seed))
    oracle_rng = MersenneTwister(seed)
    latent = fit.scales[:sd_group] .* randn(oracle_rng, 80)
    eta_fixed = predict(fit, data; type = :link)
    expected = Float64[
        rand(oracle_rng) < fit.scales[:zi][i] ? 0 :
            rand(oracle_rng, Distributions.Poisson(
                exp(eta_fixed[i] + latent[data.g[i]])))
        for i in eachindex(data.y)
    ]
    @test simulated == expected

    boot = bootstrap_result(fit; data, B = 1,
                            rng = MersenneTwister(2026102420), failures = :skip)
    @test boot.attempted == 1
    @test boot.used + boot.failed == 1
end
