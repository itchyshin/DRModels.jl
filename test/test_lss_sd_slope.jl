using DRModels
using Test
using Random
using LinearAlgebra: cholesky, Diagonal, I, logdet, Symmetric, dot

function _gaussian_sd_slope_fixture(; seed = 20261022, G = 80, m = 8)
    rng = MersenneTwister(seed)
    z_group = repeat([0.0, 1.0], inner = G ÷ 2)
    g = repeat(1:G, inner = m)
    z = z_group[g]
    x = randn(rng, G * m)
    alpha = [-0.65, 0.55]
    beta = [0.25, 0.35]
    sigma = 0.45
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    y = beta[1] .+ beta[2] .* x .+ x .* b[g] .+ sigma .* randn(rng, length(x))
    return (; y, x, z, g, z_group), alpha
end

function _gaussian_slope_dense_ml(data, beta, sigma, alpha)
    total = 0.0
    for group in unique(data.g)
        rows = findall(==(group), data.g)
        tau = exp(alpha[1] + alpha[2] * data.z[rows[1]])
        loading = data.x[rows]
        residual = data.y[rows] .- beta[1] .- beta[2] .* loading
        V = Matrix(Diagonal(fill(sigma^2, length(rows)))) .+ tau^2 .* (loading * loading')
        F = cholesky(Symmetric(V))
        total += 0.5 * (logdet(F) + dot(residual, F \ residual) +
                        length(rows) * log(2π))
    end
    return total
end

function _gaussian_slope_dense_reml(data, beta, sigma, alpha)
    n = length(data.y)
    X = hcat(ones(n), data.x)
    V = Matrix(Diagonal(fill(sigma^2, n)))
    for group in unique(data.g)
        rows = findall(==(group), data.g)
        tau = exp(alpha[1] + alpha[2] * data.z[rows[1]])
        loading = data.x[rows]
        V[rows, rows] .+= tau^2 .* (loading * loading')
    end
    F = cholesky(Symmetric(V))
    ViX = F \ X
    information = X' * ViX
    beta_gls = information \ (X' * (F \ data.y))
    residual = data.y .- X * beta_gls
    return 0.5 * (logdet(F) + logdet(cholesky(Symmetric(information))) +
                  dot(residual, F \ residual) + (n - size(X, 2)) * log(2π))
end

@testset "Gaussian LSS predictor-dependent SD for an independent slope" begin
    data, alpha = _gaussian_sd_slope_fixture()
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, coef(x)) ~ z))

    fit = drm(formula, Gaussian(); data, g_tol = 1e-7)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.30
    @test fit.nll(fit.theta) ≈ _gaussian_slope_dense_ml(
        data, coef(fit, :mu), exp(only(coef(fit, :sigma))), coef(fit, :sd)) atol = 1e-7

    predicted_sd = predict_parameters(fit, data)[:sd]
    fitted_sd = exp.(coef(fit, :sd)[1] .+ coef(fit, :sd)[2] .* data.z_group)[data.g]
    @test length(predicted_sd) == length(data.y)
    @test all(isfinite, predicted_sd)
    @test predicted_sd ≈ fitted_sd
    @test fit.scales[:sd] ≈ predicted_sd

    simulator = DRModels._lss_marginal_simulator(fit, data)
    sim_seed = 20261023
    simulated = simulator(MersenneTwister(sim_seed))
    sim_rng = MersenneTwister(sim_seed)
    random_coefficients = randn(sim_rng, length(data.z_group))
    residual_noise = randn(sim_rng, length(data.y))
    expected_simulation = fit.means[:mu] .+
        data.x .* predicted_sd .* random_coefficients[data.g] .+
        fit.scales[:sigma] .* residual_noise
    @test simulated ≈ expected_simulation
    wrong_intercept_simulation = fit.means[:mu] .+
        predicted_sd .* random_coefficients[data.g] .+
        fit.scales[:sigma] .* residual_noise
    @test maximum(abs.(simulated .- wrong_intercept_simulation)) > 0.1
    boot = bootstrap_result(fit; data, B = 2,
                            rng = MersenneTwister(20261024), failures = :skip)
    @test boot.failed == 0

    fit_reml = drm(formula, Gaussian(); data, method = :REML, g_tol = 1e-7)
    @test fit_reml.converged
    @test reml_loglik(fit_reml) ≈ -_gaussian_slope_dense_reml(
        data, coef(fit_reml, :mu), exp(only(coef(fit_reml, :sigma))),
        coef(fit_reml, :sd)) atol = 1e-6

    constant = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                  @formula(sd(g, coef(x)) ~ 1))
    fit_constant = drm(constant, Gaussian(); data, g_tol = 1e-7)
    fit_legacy = drm(bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1)),
                     Gaussian(); data, g_tol = 1e-7)
    @test fit_constant.loglik ≈ fit_legacy.loglik atol = 1e-6
    @test coef(fit_constant, :mu) ≈ coef(fit_legacy, :mu) atol = 1e-5

    missing_selector = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                          @formula(sd(g) ~ z))
    wrong_selector = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                        @formula(sd(g, coef(intercept)) ~ z))
    @test_throws ArgumentError drm(missing_selector, Gaussian(); data)
    @test_throws ArgumentError drm(wrong_selector, Gaussian(); data)
end
