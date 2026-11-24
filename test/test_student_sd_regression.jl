using DRModels, Random, Test, QuadGK, StableRNGs
import Distributions

function _student_sd_fixture(; G = 32, m = 20, alpha = [-0.65, 0.45], seed = 202610182)
    rng = StableRNG(seed)
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[group]
    x = randn(rng, length(group))
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    y = 0.4 .+ 0.3 .* x .+ b[group] .+ 0.6 .* rand(rng, Distributions.TDist(8), length(group))
    return (; y, x, z, group, z_group), alpha
end

function _student_slope_sd_fixture(; G = 60, m = 16, alpha = [-0.65, 0.45], seed = 202610186)
    rng = StableRNG(seed)
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = z_group[group]
    x = randn(rng, length(group))
    b = exp.(alpha[1] .+ alpha[2] .* z_group) .* randn(rng, G)
    y = 0.4 .+ 0.3 .* x .+ b[group] .* x .+ 0.6 .* rand(rng, Distributions.TDist(8), length(group))
    return (; y, x, z, group, z_group), alpha
end

@testset "Student random-intercept SD regression" begin
    data, alpha = _student_sd_fixture()
    formula = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                 @formula(nu ~ 1), @formula(sd(group, coef(intercept)) ~ z))
    fit = drm(formula, Student(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :nu, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.7
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(32), data.z_group) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd_group][data.group]
    predicted_sd = predict_parameters(fit, data; se = true)[:sd]
    @test all(isfinite, predicted_sd.value)
    @test all(isfinite, predicted_sd.se)
    @test_throws ArgumentError re_sd(fit)

    beta_mu = coef(fit, :mu)
    sigma_hat = exp(coef(fit, :sigma)[1])
    nu_hat = 2 + exp(coef(fit, :nu)[1])
    alpha_hat = coef(fit, :sd)
    Xmu = hcat(ones(length(data.y)), data.x)
    oracle_nll = 0.0
    for gi in 1:32
        rows = findall(==(gi), data.group)
        sdg = exp(alpha_hat[1] + alpha_hat[2] * data.z_group[gi])
        integrand = u -> begin
            ll = sum(i -> Distributions.logpdf(Distributions.TDist(nu_hat),
                (data.y[i] - (sum(Xmu[i, :] .* beta_mu) + sdg * u)) / sigma_hat) - log(sigma_hat), rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
        oracle_nll -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle_nll atol = 3e-3

    constant_sd = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                     @formula(nu ~ 1), @formula(sd(group) ~ 1))
    old_constant = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                      @formula(nu ~ 1))
    fit_constant_sd = drm(constant_sd, Student(); data)
    fit_old_constant = drm(old_constant, Student(); data)
    @test fit_constant_sd.loglik ≈ fit_old_constant.loglik atol = 1e-5
    @test coef(fit_constant_sd, :sd)[1] ≈ coef(fit_old_constant, :resd)[1] atol = 1e-4

    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(202610185),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end

@testset "Student independent random-slope SD regression" begin
    data, alpha = _student_slope_sd_fixture()
    formula = bf(@formula(y ~ x + (0 + x | group)), @formula(sigma ~ 1),
                 @formula(nu ~ 1), @formula(sd(group, coef(x)) ~ z))
    fit = drm(formula, Student(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :nu, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.65
    expected_group_sd = exp.(hcat(ones(length(data.z_group)), data.z_group) * coef(fit, :sd))
    @test fit.scales[:sd_group] ≈ expected_group_sd
    @test predict_parameters(fit, data)[:sd] ≈ expected_group_sd[data.group]
    newdata = merge(data, (; x = 2 .* data.x))
    @test predict_parameters(fit, newdata)[:sd] ≈ expected_group_sd[data.group]

    beta_mu = coef(fit, :mu)
    sigma_hat = exp(coef(fit, :sigma)[1])
    nu_hat = 2 + exp(coef(fit, :nu)[1])
    alpha_hat = coef(fit, :sd)
    Xmu = hcat(ones(length(data.y)), data.x)
    oracle_nll = 0.0
    for gi in eachindex(data.z_group)
        rows = findall(==(gi), data.group)
        sdg = exp(alpha_hat[1] + alpha_hat[2] * data.z_group[gi])
        integrand = u -> begin
            ll = sum(i -> Distributions.logpdf(Distributions.TDist(nu_hat),
                (data.y[i] - (sum(Xmu[i, :] .* beta_mu) + data.x[i] * sdg * u)) / sigma_hat) - log(sigma_hat), rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
        oracle_nll -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle_nll atol = 3e-3

    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(202610187),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end

@testset "Student SD regression rejects unsupported random-effect structures" begin
    data, _ = _student_sd_fixture(G = 12, m = 6, seed = 202610183)
    sd_formula = @formula(sd(group) ~ z)
    ordinary = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                  @formula(nu ~ 1), sd_formula)
    varying_within_group = merge(data,
        (; z = data.z .+ randn(MersenneTwister(202610184), length(data.y))))
    @test_throws ArgumentError drm(ordinary, Student(); data = varying_within_group)

    slope = bf(@formula(y ~ x + (0 + x | group)), @formula(sigma ~ 1),
               @formula(nu ~ 1), sd_formula)
    @test only(DRModels._sd_target_parts(slope)).target.member === nothing

    correlated = bf(@formula(y ~ x + (1 + x | group)), @formula(sigma ~ 1),
                    @formula(nu ~ 1), sd_formula)
    @test_throws ArgumentError drm(correlated, Student(); data)

    slope_only = bf(@formula(y ~ 1 + (0 + x | group)), @formula(sigma ~ 1),
                    @formula(nu ~ 1), @formula(sd(group, coef(x)) ~ z))
    bad_slope = merge(data, (; x = fill(NaN, length(data.x))))
    @test_throws ArgumentError drm(slope_only, Student(); data = bad_slope)

    selected = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                  @formula(nu ~ 1), @formula(sd(group, coef(x)) ~ z))
    @test_throws ArgumentError drm(selected, Student(); data)

    varying_sigma = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ z),
                       @formula(nu ~ 1), sd_formula)
    @test_throws ArgumentError drm(varying_sigma, Student(); data)

    varying_nu = bf(@formula(y ~ x + (1 | group)), @formula(sigma ~ 1),
                    @formula(nu ~ z), sd_formula)
    @test_throws ArgumentError drm(varying_nu, Student(); data)

    @test drm(ordinary, Student(); data, marginal = :Laplace).marginal === :Laplace   # laplace-parity: now fitted
end
