# Predictor-dependent SD for an independent random slope on log-sigma.
using DRModels, Test, Random, LinearAlgebra, QuadGK, Serialization
import Distributions

function _betabinomial_sigma_sd_slope_fixture(; G = 60, m = 24, trials = 24,
                                               seed = 20261029)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[group]
    x = randn(rng, length(group))
    w = randn(rng, length(group))
    alpha = [-1.2, 0.30]
    tau = exp.(alpha[1] .+ alpha[2] .* zgroup)
    b = tau .* randn(rng, G)
    eta_mu = -0.15 .+ 0.45 .* w
    log_sigma = -0.55 .+ x .* b[group]
    sigma = exp.(log_sigma)
    precision = exp.(-2 .* log_sigma)
    mu = 1 ./ (1 .+ exp.(-eta_mu))
    successes = [rand(rng, Distributions.BetaBinomial(
        trials, mu[i] * precision[i], (1 - mu[i]) * precision[i]))
        for i in eachindex(mu)]
    failures = trials .- successes
    @test rank(hcat(ones(G), zgroup)) == 2
    return (; successes = Float64.(successes), failures = Float64.(failures),
            group, z, zgroup, x, w), alpha
end

@testset "BetaBinomial sigma slope SD target validation" begin
    data, _ = _betabinomial_sigma_sd_slope_fixture(G = 8, m = 3)
    missing_selector = bf(@formula(cbind(successes, failures) ~ w),
                          @formula(sigma ~ 1 + (0 + x | group)),
                          @formula(sd(group, sigma) ~ z))
    @test_throws ArgumentError drm(missing_selector, BetaBinomial(); data, se = false)

    formula = bf(@formula(cbind(successes, failures) ~ w),
                 @formula(sigma ~ 1 + (0 + x | group)),
                 @formula(sd(group, sigma, coef(x)) ~ z))
    nonnumeric = merge(data, (; x = string.(data.x)))
    @test_throws ArgumentError drm(formula, BetaBinomial(); data = nonnumeric, se = false)
    x_nonfinite = copy(data.x)
    x_nonfinite[1] = NaN
    nonfinite = merge(data, (; x = x_nonfinite))
    @test_throws ArgumentError drm(formula, BetaBinomial(); data = nonfinite, se = false)
end

@testset "BetaBinomial sigma random-slope SD regression" begin
    data, alpha = _betabinomial_sigma_sd_slope_fixture()
    formula = bf(@formula(cbind(successes, failures) ~ w),
                 @formula(sigma ~ 1 + (0 + x | group)),
                 @formula(sd(group, sigma, coef(x)) ~ z))
    fit = drm(formula, BetaBinomial(); data, se = false)

    @test fit.converged
    @test coef(fit, :sd) ≈ alpha atol = 0.65
    sd_group = exp.(hcat(ones(length(data.zgroup)), data.zgroup) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ sd_group
    @test predict_parameters(fit, data)[:sd] ≈ sd_group[data.group]
    @test length(sigma(fit)) == length(data.group)
    @test all(isfinite, sigma(fit))
    @test sigma(fit) ≈ fit.scales[:sigma]
    @test ranef(fit)[Symbol("group_x_logsigma")] ≈ fit.scales[:sigma_random_effect]

    path = tempname()
    try
        open(path, "w") do io
            serialize(io, fit)
        end
        reloaded = open(deserialize, path)
        @test coef(reloaded, :sd) ≈ coef(fit, :sd) atol = 1e-12
        @test predict_parameters(reloaded, data)[:sd] ≈
              predict_parameters(fit, data)[:sd] atol = 1e-12
    finally
        rm(path; force = true)
    end

    # Independent groupwise Gaussian integral with each row's slope loading.
    eta_mu = hcat(ones(length(data.w)), data.w) * coef(fit, :mu)
    eta_sigma = fill(only(coef(fit, :sigma)), length(data.group))
    oracle_nll = 0.0
    for gi in eachindex(sd_group)
        rows = findall(==(gi), data.group)
        integrand(u) = begin
            ll = sum(i -> begin
                log_sigma = eta_sigma[i] + data.x[i] * sd_group[gi] * u
                phi = exp(-2 * log_sigma)
                mu_i = 1 / (1 + exp(-eta_mu[i]))
                Distributions.logpdf(Distributions.BetaBinomial(
                    24, mu_i * phi, (1 - mu_i) * phi), data.successes[i])
            end, rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-8)
        oracle_nll -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle_nll atol = 0.08

    marginal_simulator = DRModels._marginal_simulator(fit, data)
    @test marginal_simulator !== nothing
    simulated = marginal_simulator(MersenneTwister(20261030))
    @test length(simulated) == length(data.group)
    @test all(0 .<= simulated .<= 24)

    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(20261031),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.failed == 0
    @test any(row -> row.param === :sd, boot.summary)
end
