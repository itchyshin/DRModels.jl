module TestLogNormalSDSlope

using DRModels
import Distributions
using LinearAlgebra
using Random
using Test

@testset "LogNormal independent-slope SD regression" begin
    rng = MersenneTwister(913)
    G, m = 48, 18
    id = repeat(["site_$(g)" for g in 1:G], inner = m)
    x = repeat(range(-1.4, 1.4; length=m), G)
    z_group = repeat(vcat(zeros(G ÷ 2), ones(G ÷ 2)), inner = m)
    z = Float64.(z_group)
    alpha = [-0.65, 0.55]
    sigma = 0.45
    b = [randn(rng) * exp(alpha[1] + alpha[2] * z_group[(g - 1) * m + 1]) for g in 1:G]
    logy = [0.35 + 0.2 * x[i] + x[i] * b[div(i - 1, m) + 1] + sigma * randn(rng)
            for i in eachindex(x)]
    y = exp.(logy)
    data = (; y, logy, x, z, id)

    fit = drm(bf(@formula(y ~ x + (0 + x | id)), @formula(sigma ~ 1),
                 @formula(sd(id, coef(x)) ~ z)), LogNormal(); data)

    @test isfinite(loglik(fit))
    sd_estimate = coef(fit, :sd)
    @test abs(sd_estimate[1] - alpha[1]) < 0.4
    @test abs(sd_estimate[2] - alpha[2]) < 0.3
    @test length(fit.scales[:sd_group]) == G
    @test length(ranef(fit)[:id]) == G
    @test all(isfinite, predict_parameters(fit, data)[:sd])
    # Reconstruct the marginal draw independently: group-specific slope SDs
    # scale fresh standard-normal effects, then LogNormal noise is added on the
    # log-response scale.
    simulator_rng = MersenneTwister(914)
    simulated = DRModels._marginal_simulator_build(fit, data)(simulator_rng)
    oracle_rng = MersenneTwister(914)
    group_effect = fit.scales[:sd_group] .* randn(oracle_rng, G)
    eta_fixed = Vector{Float64}(predict(fit, data; type = :link))
    residual = [randn(oracle_rng) for _ in eachindex(y)]
    expected = exp.(eta_fixed .+ x .* group_effect[repeat(1:G, inner = m)] .+
                    first(fit.scales[:sigma]) .* residual)
    @test simulated ≈ expected rtol = 1e-13 atol = 1e-13

    # Independent dense covariance oracle: log(y)_g ~ N(Xβ, σ²I + τ_g² x_gx_g′).
    beta = coef(fit, :mu)
    sigma_hat = first(fit.scales[:sigma])
    direct_loglik = 0.0
    for g in 1:G
        rows = ((g - 1) * m + 1):(g * m)
        residual = logy[rows] .- beta[1] .- beta[2] .* x[rows]
        covariance = sigma_hat^2 * Matrix{Float64}(I, m, m) .+
                     fit.scales[:sd_group][g]^2 .* (x[rows] * x[rows]')
        direct_loglik += Distributions.logpdf(
            Distributions.MvNormal(zeros(m), Symmetric(covariance)), residual)
    end
    @test loglik(fit) ≈ direct_loglik - sum(log, y) atol = 1e-5

    # Conditional coefficient means use the same x-weighted sufficient statistics.
    sigma2_hat = sigma_hat^2
    blup = ranef(fit)[:id]
    for g in 1:G
        rows = ((g - 1) * m + 1):(g * m)
        residual = logy[rows] .- beta[1] .- beta[2] .* x[rows]
        C = sum(x[rows] .* residual) / sigma2_hat
        S = sum(abs2, x[rows]) / sigma2_hat
        expected = fit.scales[:sd_group][g]^2 * C / (1 + fit.scales[:sd_group][g]^2 * S)
        @test blup[g] ≈ expected atol = 1e-8
    end

    fit_constant = drm(bf(@formula(y ~ x + (0 + x | id)), @formula(sigma ~ 1),
                          @formula(sd(id, coef(x)) ~ 1)), LogNormal(); data)
    fit_base = drm(bf(@formula(y ~ x + (0 + x | id)), @formula(sigma ~ 1)),
                   LogNormal(); data)
    @test loglik(fit_constant) ≈ loglik(fit_base) atol = 1e-5
    @test only(coef(fit_constant, :sd)) ≈ only(coef(fit_base, :resd)) atol = 1e-4

    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (0 + x | id)), @formula(sigma ~ 1),
           @formula(sd(id, coef(intercept)) ~ z)), LogNormal(); data)
    varying = merge(data, (; z = randn(rng, length(y))))
    @test_throws ArgumentError drm(
        bf(@formula(y ~ x + (0 + x | id)), @formula(sigma ~ 1),
           @formula(sd(id, coef(x)) ~ z)), LogNormal(); data = varying)
end

end # module
