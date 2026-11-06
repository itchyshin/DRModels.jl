using DRModels, Random, Test, LinearAlgebra, QuadGK, Serialization
import Distributions

function _gamma_sigma_sd_fixture(; G = 64, m = 18, seed = 20261025)
    rng = MersenneTwister(seed)
    group = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[group]
    x = randn(rng, length(group))
    alpha = [-1.15, 0.35]
    tau = exp.(alpha[1] .+ alpha[2] .* zgroup)
    b = tau .* randn(rng, G)
    eta_sigma = -0.55 .+ b[group]
    mu = exp.(0.2 .+ 0.25 .* x)
    sigma = exp.(eta_sigma)
    y = [rand(rng, Distributions.Gamma(1 / sigma[i]^2,
                                       mu[i] * sigma[i]^2))
         for i in eachindex(mu)]
    return (; y = Float64.(y), x, z, zgroup, group), alpha
end

@testset "Gamma sigma random-intercept SD regression" begin
    data, alpha = _gamma_sigma_sd_fixture()
    formula = bf(@formula(y ~ 1 + x),
                 @formula(sigma ~ 1 + (1 | group)),
                 @formula(sd(group, sigma) ~ z))
    fit = drm(formula, Gamma(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.65
    @test fit.scales[:sd_group] ≈
          exp.(hcat(ones(length(data.zgroup)), data.zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈
          fit.scales[:sd_group][data.group]
    @test predict_parameters(fit, data)[:sigma] ≈
          fill(exp(only(coef(fit, :sigma))), length(data.y))
    @test sigma(fit) ≈ exp.(predict_parameters(fit, data; type = :link)[:sigma] .+
                              fit.scales[:sigma_random_effect][data.group])
    @test ranef(fit)[Symbol("group_logsigma")] ≈
          fit.scales[:sigma_random_effect]

    fixed_eta_mu = hcat(ones(length(data.x)), data.x) * coef(fit, :mu)
    fixed_eta_sigma = fill(only(coef(fit, :sigma)), length(data.y))
    group_sd = fit.scales[:sd_group]
    oracle_nll = 0.0
    for gi in eachindex(group_sd)
        rows = findall(==(gi), data.group)
        integrand(u) = begin
            ll = sum(i -> begin
                log_sigma = fixed_eta_sigma[i] + group_sd[gi] * u
                shape = exp(-2 * log_sigma)
                mu = exp(fixed_eta_mu[i])
                Distributions.logpdf(
                    Distributions.Gamma(shape, mu / shape), data.y[i])
            end, rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
        oracle_nll -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle_nll atol = 0.08

    buffer = IOBuffer()
    serialize(buffer, fit)
    seekstart(buffer)
    reloaded = deserialize(buffer)
    @test coef(reloaded, :sd) ≈ coef(fit, :sd) atol = 1e-12
    @test predict_parameters(reloaded, data)[:sd] ≈
          predict_parameters(fit, data)[:sd] atol = 1e-12
end

@testset "Gamma sigma SD regression refuses ignored covariance inputs" begin
    data, _ = _gamma_sigma_sd_fixture(G = 8, m = 4, seed = 20261026)
    formula = bf(@formula(y ~ 1 + x),
                 @formula(sigma ~ 1 + (1 | group)),
                 @formula(sd(group, sigma) ~ z))
    K = Matrix{Float64}(I, 8, 8)
    @test_throws ArgumentError drm(formula, Gamma(); data, K)
end

@testset "Gamma sigma random-slope SD regression" begin
    rng = MersenneTwister(20261027)
    G, m = 120, 18
    group = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[group]
    x = randn(rng, G * m)
    alpha = [log(0.18), 0.30]
    tau = exp.(alpha[1] .+ alpha[2] .* zgroup)
    b = tau .* randn(rng, G)
    eta_sigma = -0.55 .+ x .* b[group]
    sigma_true = exp.(eta_sigma)
    mu = exp.(0.2 .+ 0.25 .* x)
    y = [rand(rng, Distributions.Gamma(1 / sigma_true[i]^2,
                                       mu[i] * sigma_true[i]^2))
         for i in eachindex(mu)]
    data = (; y = Float64.(y), x, z, group, zgroup)
    formula = bf(@formula(y ~ 1 + x),
                 @formula(sigma ~ 1 + (0 + x | group)),
                 @formula(sd(group, sigma, coef(x)) ~ z))
    fit = drm(formula, Gamma(); data)

    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test fit.coefnames[end].second == ["(Intercept)", "z"]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.7
    @test fit.scales[:sd_group] ≈
          exp.(hcat(ones(G), zgroup) * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈
          fit.scales[:sd_group][group]
    mode = ranef(fit)[Symbol("group_x_logsigma")]
    eta_sigma_fixed = fill(only(coef(fit, :sigma)), length(y))
    @test sigma(fit) ≈ exp.(eta_sigma_fixed .+ x .* mode[group])

    beta_mu = coef(fit, :mu)
    fixed_eta_mu = hcat(ones(length(x)), x) * beta_mu
    group_sd = fit.scales[:sd_group]
    oracle_nll = 0.0
    for gi in 1:G
        rows = findall(==(gi), group)
        integrand(u) = begin
            ll = sum(i -> begin
                log_sigma = eta_sigma_fixed[i] + x[i] * group_sd[gi] * u
                shape = exp(-2 * log_sigma)
                local_mu = exp(fixed_eta_mu[i])
                Distributions.logpdf(
                    Distributions.Gamma(shape, local_mu / shape), y[i])
            end, rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
        oracle_nll -= log(value)
    end
    @test fit.nll(coef(fit)) ≈ oracle_nll atol = 0.12
end
