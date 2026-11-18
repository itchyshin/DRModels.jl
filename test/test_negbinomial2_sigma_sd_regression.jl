using DRModels, Random, Test, LinearAlgebra, QuadGK, Serialization
import Distributions

function _nb2_sigma_sd_fixture(; slope = false, G = 64, m = 18,
                                seed = slope ? 20261030 : 20261029)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    wgroup = collect(range(-1.0, 1.0; length = G))
    w = wgroup[group]
    x = randn(rng, length(group))
    loading = slope ? x : ones(length(x))
    alpha = [-0.9, 0.4]
    tau = exp.(alpha[1] .+ alpha[2] .* wgroup)
    b = tau .* randn(rng, G)
    eta_sigma = -0.6 .+ loading .* b[group]
    mu = exp.(0.35 .- 0.18 .* x)
    shape = exp.(-2 .* eta_sigma)
    y = [rand(rng, Distributions.NegativeBinomial(shape[i],
              shape[i] / (shape[i] + mu[i]))) for i in eachindex(mu)]
    return (; y = Float64.(y), x, w, wgroup, group), alpha
end

function _nb2_sigma_sd_oracle_nll(data, fit; slope::Bool)
    fixed_eta_mu = hcat(ones(length(data.x)), data.x) * coef(fit, :mu)
    fixed_eta_sigma = fill(only(coef(fit, :sigma)), length(data.y))
    group_sd = fit.scales[:sd_group]
    loading = slope ? data.x : ones(length(data.x))
    result = 0.0
    for gi in eachindex(group_sd)
        rows = findall(==(gi), data.group)
        integrand(u) = begin
            ll = sum(i -> begin
                eta_sigma = fixed_eta_sigma[i] + loading[i] * group_sd[gi] * u
                shape = exp(-2 * eta_sigma)
                mu = exp(fixed_eta_mu[i])
                Distributions.logpdf(
                    Distributions.NegativeBinomial(shape, shape / (shape + mu)),
                    data.y[i])
            end, rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
        result -= log(value)
    end
    return result
end

@testset "NB2 sigma random-intercept SD regression" begin
    data, alpha = _nb2_sigma_sd_fixture()
    formula = bf(@formula(y ~ 1 + x),
                 @formula(sigma ~ 1 + (1 | group)),
                 @formula(sd(group, sigma) ~ w))
    fit = drm(formula, NegBinomial2(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[1] - alpha[1]) < 0.45
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.45
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(data.wgroup)),
        data.wgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈
          fit.scales[:sd_group][data.group]
    @test sigma(fit) ≈ exp.(fill(only(coef(fit, :sigma)), length(data.y)) .+
        fit.scales[:sigma_random_effect][data.group])
    @test ranef(fit)[Symbol("group_logsigma")] ≈
          fit.scales[:sigma_random_effect]
    @test fit.nll(coef(fit)) ≈ _nb2_sigma_sd_oracle_nll(data, fit; slope = false) atol = 0.1

    buffer = IOBuffer()
    serialize(buffer, fit)
    seekstart(buffer)
    reloaded = deserialize(buffer)
    @test coef(reloaded, :sd) ≈ coef(fit, :sd) atol = 1e-12
    @test predict_parameters(reloaded, data)[:sd] ≈
          predict_parameters(fit, data)[:sd] atol = 1e-12
end

@testset "NB2 sigma random-slope SD regression" begin
    data, alpha = _nb2_sigma_sd_fixture(slope = true)
    formula = bf(@formula(y ~ 1 + x),
                 @formula(sigma ~ 1 + (0 + x | group)),
                 @formula(sd(group, sigma, coef(x)) ~ w))
    fit = drm(formula, NegBinomial2(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[1] - alpha[1]) < 0.45
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.45
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(data.wgroup)),
        data.wgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈
          fit.scales[:sd_group][data.group]
    @test sigma(fit) ≈ exp.(fill(only(coef(fit, :sigma)), length(data.y)) .+
        data.x .* fit.scales[:sigma_random_effect][data.group])
    @test ranef(fit)[Symbol("group_x_logsigma")] ≈
          fit.scales[:sigma_random_effect]
    @test fit.nll(coef(fit)) ≈ _nb2_sigma_sd_oracle_nll(data, fit; slope = true) atol = 0.1
end
