using DRModels, Random, Test
import Distributions

function _zeroonebeta_sd_simulation_integrity_fixture(; G = 40, m = 12,
                                                       seed = 20261022)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[group]
    x = randn(rng, length(group))
    alpha = [-0.6, 0.45]
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    mu = 1 ./ (1 .+ exp.(-(0.2 .+ 0.25 .* x .+ b[group])))
    sigma = 0.35
    phi = 1 / sigma^2
    zoi, coi = 0.22, 0.4
    y = Vector{Float64}(undef, length(group))
    for i in eachindex(y)
        if rand(rng) < zoi
            y[i] = rand(rng) < coi ? 1.0 : 0.0
        else
            y[i] = rand(rng, Distributions.Beta(mu[i] * phi,
                                               (1 - mu[i]) * phi))
        end
    end
    return (; y, x, z, group, z_group)
end

@testset "ZeroOneBeta predictor-dependent SD marginal simulation" begin
    data = _zeroonebeta_sd_simulation_integrity_fixture()
    formula = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                 @formula(zoi ~ 1), @formula(coi ~ 1),
                 @formula(sd(group, coef(intercept)) ~ z))
    fit = drm(formula, ZeroOneBeta(); data)
    @test fit.converged

    seed = 20261023
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    marginal = simulator(MersenneTwister(seed))
    repeated = simulator(MersenneTwister(seed))
    @test length(marginal) == length(data.y)
    @test all(isfinite, marginal)
    @test all((0 .<= marginal) .& (marginal .<= 1))
    @test marginal == repeated

    # Independently reconstruct the latent group effect and zero-one-Beta draw.
    oracle_rng = MersenneTwister(seed)
    latent = randn(oracle_rng, length(fit.scales[:sd_group]))
    eta_fixed = predict(fit, data; type = :link)
    eta_star = eta_fixed .+
        fit.scales[:sd_group][data.group] .* latent[data.group]
    mu_star = 1 ./ (1 .+ exp.(-eta_star))
    phi = 1 ./ fit.scales[:sigma] .^ 2
    zoi = fit.scales[:zoi]
    coi = fit.scales[:coi]
    expected = Vector{Float64}(undef, length(mu_star))
    for i in eachindex(expected)
        if rand(oracle_rng) < zoi[i]
            expected[i] = rand(oracle_rng) < coi[i] ? 1.0 : 0.0
        else
            expected[i] = rand(oracle_rng, Distributions.Beta(
                mu_star[i] * phi[i], (1 - mu_star[i]) * phi[i]))
        end
    end
    @test isapprox(marginal, expected; rtol = 1e-13, atol = 1e-13)

    boot = bootstrap_result(fit; data, B = 1,
        rng = MersenneTwister(20261024), failures = :skip,
        check_converged = true)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end
