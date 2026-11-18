# Predictor-dependent SD for one selected ZeroOneBeta mean random slope.
using DRModels, Random, Test
import Distributions
import QuadGK

function _zeroonebeta_slope_sd_fixture(alpha; G = 40, m = 30, seed = 20261021)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = 1 ./ (1 .+ exp.(-(0.15 .+ 0.3 .* x .+ x .* b[g])))
    sigma = 0.35
    phi = 1 / sigma^2
    zoi, coi = 0.22, 0.4
    y = Vector{Float64}(undef, length(g))
    for i in eachindex(y)
        if rand(rng) < zoi
            y[i] = rand(rng) < coi ? 1.0 : 0.0
        else
            y[i] = rand(rng, Distributions.Beta(mu[i] * phi, (1 - mu[i]) * phi))
        end
    end
    return (; y, x, z, zgroup, g), alpha
end

@testset "ZeroOneBeta predictor-dependent random-slope SD" begin
    data, alpha = _zeroonebeta_slope_sd_fixture([-0.55, 0.35])
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(zoi ~ 1), @formula(coi ~ 1),
                 @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, ZeroOneBeta(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :zoi, :coi, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.75
    Zg = hcat(ones(length(data.zgroup)), data.zgroup)
    expected_group_sd = exp.(Zg * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ expected_group_sd
    @test fit.scales[:sd] ≈ expected_group_sd[data.g]
    predicted = predict_parameters(fit, data)
    @test predicted[:sd] ≈ expected_group_sd[data.g]
    predicted_link = predict_parameters(fit, data; type = :link)
    @test predicted_link[:sd] ≈ log.(expected_group_sd[data.g])
    predicted_se = predict_parameters(fit, data; se = true)
    @test predicted_se[:sd].value ≈ predicted[:sd]
    @test all(isfinite, predicted_se[:sd].se)
    @test isfinite(loglik(fit))

    # Independent one-dimensional normal quadrature at the fitted coefficients.
    theta = coef(fit)
    eta_mu = hcat(ones(length(data.x)), data.x) * coef(fit, :mu)
    eta_sigma = only(coef(fit, :sigma))
    eta_zoi = only(coef(fit, :zoi))
    eta_coi = only(coef(fit, :coi))
    oracle_loglik = 0.0
    for group in 1:length(data.zgroup)
        rows = findall(==(group), data.g)
        tau = expected_group_sd[group]
        function loglik_at(u)
            contribution = 0.0
            for i in rows
                if data.y[i] == 0
                    contribution += DRModels._log_logistic(eta_zoi) +
                        DRModels._log1m_logistic(eta_coi)
                elseif data.y[i] == 1
                    contribution += DRModels._log_logistic(eta_zoi) +
                        DRModels._log_logistic(eta_coi)
                else
                    eta = clamp(eta_mu[i] + data.x[i] * tau * u, -30.0, 30.0)
                    mu = 1 / (1 + exp(-eta))
                    phi_i = exp(-2 * eta_sigma)
                    contribution += DRModels._log1m_logistic(eta_zoi) +
                        Distributions.logpdf(Distributions.Beta(mu * phi_i,
                            (1 - mu) * phi_i), data.y[i])
                end
            end
            return contribution
        end
        grid = range(-8.0, 8.0; length = 161)
        shift = maximum(loglik_at(u) + Distributions.logpdf(Distributions.Normal(), u)
                        for u in grid)
        integral, _ = QuadGK.quadgk(u -> exp(loglik_at(u) +
            Distributions.logpdf(Distributions.Normal(), u) - shift), -9.0, 9.0;
            rtol = 1e-9)
        oracle_loglik += shift + log(integral)
    end
    @test isapprox(fit.nll(theta), -oracle_loglik; atol = 2e-5)
end
