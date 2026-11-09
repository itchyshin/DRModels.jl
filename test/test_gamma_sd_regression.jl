using DRModels, Random, Test, QuadGK
import Distributions

function _gamma_sd_fixture(alpha; G = 40, m = 20, seed = 202610181)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = exp.(0.25 .+ 0.2 .* x .+ b[g])
    sigma = 0.35
    shape = 1 / sigma^2
    y = [rand(rng, Distributions.Gamma(shape, v / shape)) for v in mu]
    return (; y = Float64.(y), x, z, g), alpha
end

function _gamma_sd_slope_fixture(alpha; G = 80, m = 25, seed = 202610188)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = exp.(0.4 .+ 0.3 .* x .+ b[g] .* x)
    sigma = 0.3
    shape = 1 / sigma^2
    y = [rand(rng, Distributions.Gamma(shape, v / shape)) for v in mu]
    return (; y = Float64.(y), x, z, g, zgroup), alpha
end

@testset "Gamma random-effect SD regression refuses unsupported designs" begin
    data, _ = _gamma_sd_fixture([-0.5, 0.3]; G = 12, m = 6,
                                seed = 202610184)
    base = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
              @formula(sd(g) ~ z))
    @test_throws ArgumentError drm(base, Gamma(); data = merge(data,
        (; z = data.z .+ randn(MersenneTwister(202610185), length(data.z)))))

    varying_sigma = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ z),
                       @formula(sd(g) ~ z))
    @test_throws ArgumentError drm(varying_sigma, Gamma(); data)

    correlated_slope = bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1),
                          @formula(sd(g) ~ z))
    @test_throws ArgumentError drm(correlated_slope, Gamma(); data)

    # `marginal = :Laplace` now fits this shape (test_laplace_parity.jl); a shape it does not cover
    # (here a correlated slope with an `sd()` formula) names the default integrator in its refusal.
    @test drm(base, Gamma(); data, marginal = :Laplace).marginal === :Laplace
    laplace_error = try
        drm(correlated_slope, Gamma(); data, marginal = :Laplace)
        nothing
    catch err
        err
    end
    @test occursin("marginal = :LA", sprint(showerror, laplace_error))

    unsupported_family = try
        drm(base, Beta(); data)
        nothing
    catch err
        err
    end
    @test unsupported_family isa ArgumentError
    @test occursin("Beta", sprint(showerror, unsupported_family))

    wrong_dpar = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                    @formula(sd(g) ~ z))
    sd_key = only(keys(wrong_dpar.sd_targets))
    wrong_dpar.sd_targets[sd_key] = merge(wrong_dpar.sd_targets[sd_key],
                                           (; dpar = :sigma))
    @test_throws ArgumentError drm(wrong_dpar, Gamma(); data)
end

@testset "Gamma random-effect SD regression: ordinary random intercept" begin
    data, alpha = _gamma_sd_fixture([-0.55, 0.35])
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, coef(intercept)) ~ z))
    fit = drm(formula, Gamma(); data)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test length(coef(fit, :sd)) == 2
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.65
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈ log.(fit.scales[:sd])
    pred = predict_parameters(fit, data; se = true)[:sd]
    @test all(isfinite, pred.se)

    beta_mu = coef(fit, :mu)
    beta_sigma = coef(fit, :sigma)
    alpha_hat = coef(fit, :sd)
    Xmu = hcat(ones(length(data.y)), data.x)
    group_z = repeat([-1.0, 1.0], inner = 20)
    group_index = data.g
    Zg = hcat(ones(40), group_z)
    eta_mu = Xmu * beta_mu
    eta_sigma = fill(beta_sigma[1], length(data.y))
    shape = exp.(-2 .* eta_sigma)
    sdgroup = exp.(Zg * alpha_hat)
    oracle = 0.0
    for group in 1:40
        rows = findall(==(group), group_index)
        f(u) = begin
            b = sdgroup[group] * u
            ll = sum(i -> Distributions.logpdf(
                Distributions.Gamma(shape[i], exp(clamp(eta_mu[i] + b, -30, 30)) / shape[i]),
                data.y[i]), rows)
            exp(ll + Distributions.logpdf(Distributions.Normal(), u))
        end
        integral, _ = QuadGK.quadgk(f, -10.0, 10.0; rtol = 1e-9)
        oracle -= log(integral)
    end
    @test fit.nll(coef(fit)) ≈ oracle atol = 2e-3

    constant = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                  @formula(sd(g) ~ 1))
    # Same node count as the plain `(1 | g)` route, so only the likelihood assembly is compared.
    f_sd = DRModels._drm_gamma_sd_regression(constant, Gamma(); data,
                                                nAGQ = DRModels._RANEF1D_AGHQ_K)
    f_old = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gamma(); data)
    @test f_sd.loglik ≈ f_old.loglik atol = 1e-5
    @test coef(f_sd, :mu) ≈ coef(f_old, :mu) atol = 1e-4
    @test coef(f_sd, :sigma) ≈ coef(f_old, :sigma) atol = 1e-4
    @test coef(f_sd, :sd)[1] ≈ coef(f_old, :resd)[1] atol = 1e-4
end

@testset "Gamma random-slope SD regression: public fit and integral" begin
    data, alpha = _gamma_sd_slope_fixture([-0.75, 0.5])
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ z))
    outcome = try
        drm(formula, Gamma(); data)
    catch err
        err
    end
    @test hasproperty(outcome, :converged)
    if hasproperty(outcome, :converged)
        fit = outcome
        @test fit.converged
        @test length(coef(fit, :sd)) == 2
        @test coef(fit, :sd)[2] > 0
        @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
        group_sd = exp.(hcat(ones(80), data.zgroup) * coef(fit, :sd))
        @test fit.scales[:sd_group] ≈ group_sd
        @test predict_parameters(fit, data)[:sd] ≈ group_sd[data.g]

        theta = [0.4, 0.3, log(0.3), alpha...]
        shape = 1 / 0.3^2
        normal = Distributions.Normal()
        oracle_nll = 0.0
        for gi in 1:80
            rows = findall(==(gi), data.g)
            sdg = exp(alpha[1] + alpha[2] * data.zgroup[gi])
            integrand = u -> begin
                eta = 0.4 .+ 0.3 .* data.x[rows] .+ sdg * u .* data.x[rows]
                ll = sum(j -> Distributions.logpdf(
                    Distributions.Gamma(shape, exp(clamp(eta[j], -30, 30)) / shape),
                    data.y[rows[j]]), eachindex(rows))
                exp(ll + Distributions.logpdf(normal, u))
            end
            value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-9)
            oracle_nll -= log(value)
        end
        @test fit.nll(theta) ≈ oracle_nll atol = 0.05
    end
end

@testset "Gamma SD-regression bootstrap retains quadrature settings" begin
    data, _ = _gamma_sd_fixture([-0.5, 0.3]; G = 24, m = 12,
                                seed = 202610182)
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ z))
    fit = drm(formula, Gamma(); data)
    @test only(fit.scales[:aghq_nodes]) == DRModels._SD_IID_AGHQ_K
    @test !haskey(DRModels._bootstrap_refit_kwargs(fit), :nAGQ)
    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(202610183),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end
