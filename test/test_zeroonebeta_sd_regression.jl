# Predictor-dependent SD for an ordinary ZeroOneBeta mean random intercept.
using DRModels, Random, Test
import Distributions

function _zeroonebeta_sd_fixture(alpha; G = 48, m = 30, seed = 20261020)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = 1 ./ (1 .+ exp.(-(0.2 .+ 0.25 .* x .+ b[g])))
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

@testset "ZeroOneBeta predictor-dependent random-intercept SD" begin
    data, alpha = _zeroonebeta_sd_fixture([-0.55, 0.4])
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(zoi ~ 1), @formula(coi ~ 1),
                 @formula(sd(g, coef(intercept)) ~ z))
    fit = drm(formula, ZeroOneBeta(); data)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :zoi, :coi, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.75
    Zg = hcat(ones(length(data.zgroup)), data.zgroup)
    @test fit.scales[:sd_group] ≈ exp.(Zg * coef(fit, :sd))
    @test Int.(fit.scales[:sd_group_index]) == data.g
    @test isfinite(loglik(fit))

    # The fitted SD target must survive public prediction on both link and
    # response scales, including subsetted/reordered prediction rows.
    eta_group = Zg * coef(fit, :sd)
    pred_link = predict_parameters(fit, data; type = :link)
    pred_response = predict_parameters(fit, data; type = :response)
    @test haskey(pred_link, :sd)
    if haskey(pred_link, :sd) && haskey(pred_response, :sd)
        @test pred_link[:sd] ≈ eta_group[data.g]
        @test pred_response[:sd] ≈ exp.(eta_group[data.g])
        pred_se = predict_parameters(fit, data; type = :response, se = true)
        @test pred_se[:sd].value ≈ exp.(eta_group[data.g])
        @test all(isfinite, pred_se[:sd].se)
        rows = [61, 1, 31, 61]
        nd = (y = data.y[rows], x = data.x[rows], z = data.z[rows], g = data.g[rows])
        pred_subset = predict_parameters(fit, nd; type = :response)
        @test pred_subset[:sd] ≈ exp.(eta_group[nd.g])
    end

    constant_formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                          @formula(zoi ~ 1), @formula(coi ~ 1),
                          @formula(sd(g, coef(intercept)) ~ 1))
    legacy_formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                        @formula(zoi ~ 1), @formula(coi ~ 1))
    fit_sd_constant = drm(constant_formula, ZeroOneBeta(); data)
    fit_legacy = drm(legacy_formula, ZeroOneBeta(); data)
    @test fit_sd_constant.loglik ≈ fit_legacy.loglik atol = 1e-5
    @test coef(fit_sd_constant, :mu) ≈ coef(fit_legacy, :mu) atol = 1e-4
    @test coef(fit_sd_constant, :sigma) ≈ coef(fit_legacy, :sigma) atol = 1e-4
    @test coef(fit_sd_constant, :zoi) ≈ coef(fit_legacy, :zoi) atol = 1e-4
    @test coef(fit_sd_constant, :coi) ≈ coef(fit_legacy, :coi) atol = 1e-4
    @test only(coef(fit_sd_constant, :sd)) ≈ only(coef(fit_legacy, :resd)) atol = 1e-4
end
