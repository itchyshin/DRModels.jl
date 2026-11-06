# Predictor-dependent SD for a BetaBinomial random intercept on sigma.
using DRModels, Test, Random, LinearAlgebra, QuadGK, Serialization
import Distributions

function _betabinomial_sigma_sd_fixture(; G = 80, m = 18, trials = 24,
                                         seed = 20261024)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[group]
    x = randn(rng, length(group))
    alpha = [-1.25, 0.35]
    tau = exp.(alpha[1] .+ alpha[2] .* zgroup)
    log_sigma = -0.55 .+ tau[group] .* randn(rng, G)[group]
    sigma = exp.(log_sigma)
    mu = 1 ./ (1 .+ exp.(-(-0.15 .+ 0.45 .* x)))
    precision = exp.(-2 .* log_sigma)
    successes = [rand(rng, Distributions.BetaBinomial(
        trials, mu[i] * precision[i], (1 - mu[i]) * precision[i]))
        for i in eachindex(mu)]
    failures = trials .- successes
    @test rank(hcat(ones(G), zgroup)) == 2
    return (; successes = Float64.(successes), failures = Float64.(failures),
            group, z, zgroup, x), alpha
end

@testset "BetaBinomial sigma random-intercept SD regression" begin
    data, alpha = _betabinomial_sigma_sd_fixture()
    formula = bf(@formula(cbind(successes, failures) ~ x),
                 @formula(sigma ~ 1 + (1 | group)),
                 @formula(sd(group, sigma) ~ z))
    fit = drm(formula, BetaBinomial(); data, se = false)

    @test fit.converged
    @test coef(fit, :sd) ≈ alpha atol = 0.65
    @test fit.scales[:sd_group] ≈
          exp.(hcat(ones(length(data.zgroup)), data.zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd_group][data.group]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈
          log.(fit.scales[:sd_group][data.group])
    @test length(sigma(fit)) == length(data.group)
    @test all(isfinite, sigma(fit))
    @test sigma(fit) ≈ fit.scales[:sigma]
    @test ranef(fit)[Symbol("group_logsigma")] ≈ fit.scales[:sigma_random_effect]
    @test predict_parameters(fit, data)[:sigma] ≈
          fill(exp(only(coef(fit, :sigma))), length(data.group))
    @test !isapprox(predict_parameters(fit, data)[:sigma], sigma(fit); atol = 1e-8)

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

    # Independent standard-normal integral of the fitted sigma-axis model.
    eta_mu = hcat(ones(length(data.x)), data.x) * coef(fit, :mu)
    eta_sigma = fill(only(coef(fit, :sigma)), length(data.group))
    group_sd = fit.scales[:sd_group]
    oracle_nll = 0.0
    for gi in eachindex(group_sd)
        rows = findall(==(gi), data.group)
        integrand(u) = begin
            ll = sum(i -> begin
                log_sigma = eta_sigma[i] + group_sd[gi] * u
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
    simulated = marginal_simulator(MersenneTwister(20261025))
    @test length(simulated) == length(data.group)
    @test all(0 .<= simulated .<= 24)

    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(20261026),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.failed == 0
    @test any(row -> row.param === :sd, boot.summary)
end
