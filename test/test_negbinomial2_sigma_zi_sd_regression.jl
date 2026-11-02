using DRModels, Random, Test, LinearAlgebra, QuadGK
import Distributions

_nb2_test_logaddexp(a, b) = max(a, b) + log1p(exp(-abs(a - b)))

function _nb2_zi_sigma_sd_fixture(; G = 64, m = 22, seed = 20261031)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    wgroup = collect(range(-1.0, 1.0; length = G))
    w = wgroup[group]
    x = randn(rng, length(group))
    alpha = [-0.75, 0.45]
    beta_zi = -1.1
    tau = exp.(alpha[1] .+ alpha[2] .* wgroup)
    b = tau .* randn(rng, G)
    eta_sigma = -0.55 .+ b[group]
    mu = exp.(0.35 .- 0.18 .* x)
    zi = 1 / (1 + exp(-beta_zi))
    shape = exp.(-2 .* eta_sigma)
    y = [rand(rng) < zi ? 0 : rand(rng,
        Distributions.NegativeBinomial(shape[i], shape[i] / (shape[i] + mu[i])))
        for i in eachindex(mu)]
    return (; y = Float64.(y), x, w, wgroup, group), alpha, beta_zi
end

function _nb2_zi_sigma_sd_oracle_nll(data, fit)
    fixed_eta_mu = hcat(ones(length(data.x)), data.x) * coef(fit, :mu)
    fixed_eta_sigma = fill(only(coef(fit, :sigma)), length(data.y))
    eta_zi = only(coef(fit, :zi))
    log_zi = -log1p(exp(-eta_zi))
    log_one_minus_zi = -log1p(exp(eta_zi))
    group_sd = fit.scales[:sd_group]
    result = 0.0
    for gi in eachindex(group_sd)
        rows = findall(==(gi), data.group)
        integrand(u) = begin
            ll = sum(i -> begin
                eta_sigma = fixed_eta_sigma[i] + group_sd[gi] * u
                shape = exp(-2 * eta_sigma)
                mu = exp(fixed_eta_mu[i])
                log_nb = Distributions.logpdf(
                    Distributions.NegativeBinomial(shape, shape / (shape + mu)),
                    data.y[i])
                data.y[i] == 0 ?
                    _nb2_test_logaddexp(log_zi, log_one_minus_zi + log_nb) :
                    log_one_minus_zi + log_nb
            end, rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
        result -= log(value)
    end
    return result
end

@testset "Zero-inflated NB2 sigma random-intercept SD regression" begin
    data, alpha, beta_zi = _nb2_zi_sigma_sd_fixture()
    formula = bf(@formula(y ~ 1 + x),
                 @formula(sigma ~ 1 + (1 | group)),
                 @formula(zi ~ 1),
                 @formula(sd(group, sigma) ~ w))
    fit = drm(formula, NegBinomial2(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :zi, :sd]
    @test abs(coef(fit, :sd)[1] - alpha[1]) < 0.45
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.45
    @test abs(only(coef(fit, :zi)) - beta_zi) < 0.45
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(data.wgroup)),
        data.wgroup) * coef(fit, :sd))
    @test fit.scales[:zi] ≈ fill(1 / (1 + exp(-only(coef(fit, :zi)))), length(data.y))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd_group][data.group]
    @test sigma(fit) ≈ exp.(fill(only(coef(fit, :sigma)), length(data.y)) .+
        fit.scales[:sigma_random_effect][data.group])
    @test fit.nll(coef(fit)) ≈ _nb2_zi_sigma_sd_oracle_nll(data, fit) atol = 0.1
end

@testset "Zero-inflated NB2 sigma SD keeps the native q1 admission boundary" begin
    data, _, _ = _nb2_zi_sigma_sd_fixture(G = 12, m = 6)
    varying_zi = bf(@formula(y ~ 1 + x),
                    @formula(sigma ~ 1 + (1 | group)),
                    @formula(zi ~ w),
                    @formula(sd(group, sigma) ~ w))
    @test_throws ArgumentError drm(varying_zi, NegBinomial2(); data)

    sigma_slope = bf(@formula(y ~ 1 + x),
                     @formula(sigma ~ 1 + (0 + x | group)),
                     @formula(zi ~ 1),
                     @formula(sd(group, sigma, coef(x)) ~ w))
    @test_throws ArgumentError drm(sigma_slope, NegBinomial2(); data)

    sigma_predictor = bf(@formula(y ~ 1 + x),
                         @formula(sigma ~ w + (1 | group)),
                         @formula(zi ~ 1),
                         @formula(sd(group, sigma) ~ w))
    @test_throws ArgumentError drm(sigma_predictor, NegBinomial2(); data)
end
