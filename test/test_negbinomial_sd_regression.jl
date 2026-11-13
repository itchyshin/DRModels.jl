# NB2 random-intercept SD regression: b_g ~ N(0, exp(2 Z_g alpha)).
using DRModels, Test, Random, QuadGK
import Distributions

function _nb2_sd_fixture(alpha; G = 40, m = 16, seed = 20261018)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = exp.(0.35 .+ 0.25 .* x .+ b[g])
    sigma = 0.42                         # log sigma is fixed at log(0.42)
    size = 1 / sigma^2
    y = [rand(rng, Distributions.NegativeBinomial(size, size / (size + v))) for v in mu]
    return (; y = Float64.(y), x, z, g), alpha
end

function _nb2_sd_slope_fixture(alpha; G = 48, m = 20, seed = 2026101819)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    for gi in 1:G
        rows = findall(==(gi), g)
        x[rows] .-= sum(x[rows]) / length(rows)
    end
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = exp.(0.35 .+ 0.25 .* x .+ b[g] .* x)
    sigma = 0.42
    size = 1 / sigma^2
    y = [rand(rng, Distributions.NegativeBinomial(size, size / (size + v))) for v in mu]
    return (; y = Float64.(y), x, z, g), alpha
end

@testset "NB2 latent SD regression: ordinary random intercept" begin
    data, alpha = _nb2_sd_fixture([-0.55, 0.35])
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, coef(intercept)) ~ z))
    fit = drm(formula, NegBinomial2(); data)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test length(coef(fit, :sd)) == 2
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    Zg = hcat(ones(40), repeat([-1.0, 1.0], inner = 20))
    @test fit.scales[:sd_group] ≈ exp.(Zg * coef(fit, :sd))
    @test Int.(fit.scales[:sd_group_index]) == data.g
    @test only(fit.scales[:aghq_nodes]) == DRModels._SD_IID_AGHQ_K
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test predict_parameters(fit, data; type = :link)[:sd] ≈ log.(fit.scales[:sd])
    pred_sd = predict_parameters(fit, data; se = true)[:sd]
    pred_sd_link = predict_parameters(fit, data; type = :link, se = true)[:sd]
    @test length(pred_sd.value) == length(data.y)
    @test length(pred_sd.se) == length(data.y)
    @test all(isfinite, pred_sd.se) && all(isfinite, pred_sd_link.se)
    @test pred_sd.se ≈ pred_sd_link.se .* pred_sd.value

    theta = copy(coef(fit))
    threaded = fetch.((Threads.@spawn(fit.nll(theta)), Threads.@spawn(fit.nll(theta))))
    @test threaded[1] ≈ threaded[2] atol = 1e-10

    constant = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                  @formula(sd(g) ~ 1))
    # Same node count as the plain `(1 | g)` route, so only the likelihood assembly is compared.
    f_sd = drm(constant, NegBinomial2(); data, nAGQ = DRModels._RANEF1D_AGHQ_K)
    f_old = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)),
                NegBinomial2(); data)
    @test f_sd.converged && f_old.converged
    @test f_sd.loglik ≈ f_old.loglik atol = 1e-5
    @test coef(f_sd, :mu) ≈ coef(f_old, :mu) atol = 1e-4
    @test coef(f_sd, :sigma) ≈ coef(f_old, :sigma) atol = 1e-4
    @test coef(f_sd, :sd)[1] ≈ coef(f_old, :resd)[1] atol = 1e-4
end

@testset "NB2 latent SD regression: independent random slope" begin
    data, alpha = _nb2_sd_slope_fixture([-0.55, 0.38])
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ z))
    fit = drm(formula, NegBinomial2(); data)
    @test fit.converged
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.6
    Zg = hcat(ones(48), repeat([-1.0, 1.0], inner = 24))
    @test fit.scales[:sd_group] ≈ exp.(Zg * coef(fit, :sd))
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]

    small, _ = _nb2_sd_slope_fixture([-0.5, 0.3]; G = 8, m = 6,
                                     seed = 2026101820)
    small_formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                       @formula(sd(g) ~ z))
    small_fit = drm(small_formula, NegBinomial2(); data = small, se = false,
                    nAGQ = 31)
    theta = [0.35, 0.25, log(0.42), -0.5, 0.3]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:8
        rows = findall(==(gi), small.g)
        tau = exp(theta[4] + theta[5] * small.z[first(rows)])
        integrand = u -> begin
            ll = sum(rows) do i
                mu = exp(theta[1] + theta[2] * small.x[i] +
                         small.x[i] * tau * u)
                Distributions.logpdf(
                    Distributions.NegativeBinomial(exp(-2 * theta[3]),
                        exp(-2 * theta[3]) / (exp(-2 * theta[3]) + mu)),
                    round(Int, small.y[i]))
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-11)
        reference += log(value)
    end
    @test isapprox(-small_fit.nll(theta), reference; atol = 2e-3)
end

@testset "NB2 SD preserves quadrature settings through bootstrap refits" begin
    data, _ = _nb2_sd_fixture([-0.5, 0.3]; G = 24, m = 12, seed = 202610184)
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ z))
    fit = drm(formula, NegBinomial2(); data, nAGQ = 3)
    @test only(fit.scales[:aghq_nodes]) == 3
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 3
    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(202610185),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end

@testset "NB2 latent SD regression: both coefficient signs recover" begin
    for (slope, seed) in ((0.38, 202610181), (-0.38, 202610182))
        data, alpha = _nb2_sd_fixture([-0.55, slope]; G = 48, m = 20, seed)
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                     @formula(sd(g) ~ z)), NegBinomial2(); data)
        @test fit.converged
        @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
        @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.6
    end
end

@testset "NB2 latent SD regression refuses unsupported routes" begin
    data, _ = _nb2_sd_fixture([-0.5, 0.3]; G = 8, m = 5)
    route(f; kwargs...) = drm(f, NegBinomial2(); data, kwargs...)
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ x), @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ (1 | g)), @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 + x | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g) + (1 | h)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + phylo(1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z), @formula(zi ~ 1)))
    animal_formula = bf(@formula(y ~ x + animal(1 | g)),
        @formula(sigma ~ 1), @formula(sd(g, animal) ~ z))
    A = [Float64(i == j) for i in 1:8, j in 1:8]
    @test_throws ArgumentError route(animal_formula; A, nAGQ = 7)
    @test_throws ArgumentError route(bf(@formula(y ~ x + offset(log_exposure) + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)); method = :REML)
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)); marginal = :VA)
    # `marginal = :Laplace` with one `(1 | g)` and `sd(g) ~ z` is a shape the Laplace-parity work fits (drmTMB's objective);
    # its agreement with drmTMB 0.8.1 is pinned by the cell nb2|ordinary|lab0|q1|sd-z|cm0|ML in test_laplace_parity.jl.
    lapfit = route(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ z)); marginal = :Laplace)
    @test lapfit.marginal === :Laplace

    varying = merge(data, (; z = copy(data.z)))
    varying.z[1] = -varying.z[1]
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)), NegBinomial2(); data = varying)
    deficient = merge(data, (; z = zeros(length(data.y))))
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)), NegBinomial2(); data = deficient)
    missing_y = merge(data, (; y = Union{Missing,Float64}[data.y[1:end-1]...; missing]))
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)), NegBinomial2(); data = missing_y)
end

@testset "NB2 latent SD regression: fixed-parameter marginal likelihood" begin
    data, _ = _nb2_sd_fixture([-0.5, 0.3]; G = 10, m = 5)
    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ z)), NegBinomial2(); data, se = false)
    theta = [0.35, 0.25, log(0.42), -0.5, 0.3]
    reference = 0.0
    normal = Distributions.Normal()
    for gi in 1:10
        rows = findall(==(gi), data.g)
        sdg = exp(theta[4] + theta[5] * data.z[first(rows)])
        size = exp(-2 * theta[3])
        integrand = u -> begin
            ll = sum(rows) do i
                mu = exp(theta[1] + theta[2] * data.x[i] + sdg * u)
                Distributions.logpdf(
                    Distributions.NegativeBinomial(size, size / (size + mu)),
                    round(Int, data.y[i]))
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-11)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-3)
end

@testset "NB2 SD marginal clamps the combined mean predictor" begin
    data = (; y = fill(Float64(round(Int, exp(20))), 12), x = repeat([-1.0, 1.0], 6),
            g = repeat(1:2, inner = 6))
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ 1))
    rhs = Dict(formula.forms)
    fixed_mu, _, _, _ = DRModels._split_ranef(rhs[:mu])
    y, Xmu, nmmu = DRModels._design(formula.response, fixed_mu, data)
    _, Xsigma, nmsigma = DRModels._design(formula.response, rhs[:sigma], data)
    gidx, G = DRModels._group_index(data.g)
    sd_rhs = only(DRModels._sd_target_parts(formula)).rhs
    Zg, nmsd = DRModels._sd_group_design(formula.response, sd_rhs, data,
                                          gidx, G, :g)
    # The extreme fixed predictor is a clamp-order control, not a default-node
    # accuracy test. Use a dense AGHQ rule here to isolate that semantic check.
    fit = DRModels._fit_negbin2_sd_regression(NegBinomial2(), y, Xmu,
        Xsigma, Zg, gidx, G, nmmu, nmsigma, nmsd, :g, 1e-8; K = 61, se = false)
    theta = [21.0, 0.0, log(0.42), 0.0]
    size = exp(-2 * theta[3])
    rule = DRModels._AGHQRule(1, 61)
    members = [findall(==(gi), data.g) for gi in 1:2]
    function objective(eta0)
        total = 0.0
        for rows in members
            ll = (i, eta) -> DRModels._nb2_logpmf(size,
                exp(clamp(eta, -20.0, 20.0)), Int(data.y[i]))
            value, _ = DRModels._aghq_group_logint(ll, rows, eta0,
                ones(length(data.y), 1), ones(1, 1), rule, [0.0])
            total += value
        end
        return -total
    end
    eta_fixed = Xmu * theta[1:2]
    correct = objective(eta_fixed)
    # Reproduce the pre-clamp construction under the same adaptive 61-node
    # quadrature rule. The distinction is that this clamps X*beta before b is
    # added, so b cannot offset the fixed predictor as it can in the model.
    preclamped = objective(DRModels._softclamp.(eta_fixed, -20.0, 20.0, 3.0))
    @test isapprox(fit.nll(theta), correct; atol = 1e-10)
    @test abs(fit.nll(theta) - preclamped) > 0.01
end
