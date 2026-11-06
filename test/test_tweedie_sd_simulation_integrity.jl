using DRModels, Test, Random
import Distributions

function _tweedie_sd_simulation_integrity_fixture(; G = 40, m = 8,
                                                  seed = 20261022)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[group]
    x = rand(rng, length(group)) .* 2 .- 1
    alpha = [-0.8, 0.65]
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    mu = exp.(0.4 .+ 0.25 .* x .+ b[group])
    sigma = 0.7
    power = 1.5
    y = map(mu) do mu_i
        phi = sigma^2
        lambda = mu_i^(2 - power) / (phi * (2 - power))
        count = rand(rng, Distributions.Poisson(lambda))
        count == 0 && return 0.0
        shape = count * (2 - power) / (power - 1)
        scale = phi * (power - 1) * mu_i^(power - 1)
        rand(rng, Distributions.Gamma(shape, scale))
    end
    return (; y, x, z, group)
end

@testset "Tweedie predictor-dependent SD simulation and bootstrap" begin
    data = _tweedie_sd_simulation_integrity_fixture()
    formula = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                 @formula(nu ~ 1), @formula(sd(group) ~ z))
    fit = drm(formula, Tweedie(); data, g_tol = 1e-5)
    @test fit.converged

    seed = 20261023
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    marginal = simulator(MersenneTwister(seed))
    repeated = simulator(MersenneTwister(seed))
    @test length(marginal) == length(data.y)
    @test all(isfinite, marginal)
    @test all(>=(0), marginal)
    @test marginal == repeated

    # Independent seeded oracle for the predictor-dependent group SD draw.
    oracle_rng = MersenneTwister(seed)
    latent = randn(oracle_rng, length(fit.scales[:sd_group]))
    eta_fixed = predict(fit, data; type = :link)
    mu_star = exp.(eta_fixed .+
        fit.scales[:sd_group][data.group] .* latent[data.group])
    expected = map(eachindex(mu_star)) do i
        phi = fit.scales[:sigma][i]^2
        power = fit.scales[:nu][i]
        lambda = mu_star[i]^(2 - power) / (phi * (2 - power))
        count = rand(oracle_rng, Distributions.Poisson(lambda))
        count == 0 && return 0.0
        shape = count * (2 - power) / (power - 1)
        scale = phi * (power - 1) * mu_star[i]^(power - 1)
        rand(oracle_rng, Distributions.Gamma(shape, scale))
    end
    @test isapprox(marginal, expected; rtol = 1e-14, atol = 1e-14)

    boot = bootstrap_result(fit; data, B = 1,
        rng = MersenneTwister(20261024), failures = :skip,
        check_converged = true)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end
