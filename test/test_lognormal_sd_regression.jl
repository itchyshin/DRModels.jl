module TestLogNormalSDRegression

using DRModels
import Distributions
using LinearAlgebra
using Random
using Test

@testset "LogNormal random-intercept SD regression" begin
    rng = MersenneTwister(418)
    G, m = 48, 24
    id = repeat(["site_$(g)" for g in 1:G], inner = m)
    z_group = repeat(vcat(zeros(G ÷ 2), ones(G ÷ 2)), inner = m)
    z = Float64.(z_group)
    x = randn(rng, G * m)
    alpha = [-0.55, 0.5]
    sigma = 0.65
    b = [randn(rng) * exp(alpha[1] + alpha[2] * z_group[(g - 1) * m + 1]) for g in 1:G]
    logy = [0.4 + 0.3 * x[i] + b[div(i - 1, m) + 1] + sigma * randn(rng) for i in eachindex(x)]
    y = exp.(logy)
    data = (; y, x, z, id)

    fit = drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1), @formula(sd(id) ~ z)),
              DRModels.LogNormal(); data)

    @test isfinite(loglik(fit))
    @test coef(fit, :sd)[2] > 0
    @test length(fit.scales[:sd_group]) == G
    @test fit.scales[:sd] ≈ fit.scales[:sd_group][round.(Int, fit.scales[:sd_group_index])]
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    simulator = DRModels._marginal_simulator_build(fit, data)
    @test simulator !== nothing
    @test all(isfinite, simulator(MersenneTwister(419)))

    fit_constant = drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1), @formula(sd(id) ~ 1)),
                       DRModels.LogNormal(); data)
    fit_base = drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1)), DRModels.LogNormal(); data)
    @test loglik(fit_constant) ≈ loglik(fit_base) atol = 1e-5
    @test coef(fit_constant, :sd)[1] ≈ coef(fit_base, :resd)[1] atol = 1e-4

    transformed = merge(data, (; logy))
    fit_gaussian = drm(bf(@formula(logy ~ x + (1 | id)), @formula(sigma ~ 1),
                          @formula(sd(id) ~ z)), Gaussian(); data = transformed)
    @test loglik(fit) ≈ loglik(fit_gaussian) - sum(log, y) atol = 1e-5

    beta = coef(fit, :mu)
    sigma_hat = fit.scales[:sigma][1]
    direct_loglik = 0.0
    for g in 1:G
        rows = ((g - 1) * m + 1):(g * m)
        residual = logy[rows] .- beta[1] .- beta[2] .* x[rows]
        covariance = sigma_hat^2 * Matrix{Float64}(I, m, m) .+
                     fit.scales[:sd_group][g]^2 .* ones(m, m)
        direct_loglik += Distributions.logpdf(
            Distributions.MvNormal(zeros(m), Symmetric(covariance)), residual)
    end
    @test loglik(fit) ≈ direct_loglik - sum(log, y) atol = 1e-5

    varying = merge(data, (; z = randn(rng, G * m)))
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1), @formula(sd(id) ~ z)),
        DRModels.LogNormal(); data = varying)

    incomplete = merge(data, (; y = [NaN; y[2:end]]))
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1), @formula(sd(id) ~ z)),
        DRModels.LogNormal(); data = incomplete)
end

end # module
