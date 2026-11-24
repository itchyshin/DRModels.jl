# Predictor-dependent random-slope SD for zero-truncated Poisson.
using DRModels, Test, Random, QuadGK
import Distributions

function _truncated_poisson_sd_slope_fixture(alpha; G = 48, m = 24,
                                              seed = 20261028)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = collect(range(-1.0, 1.0; length = G))
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    lambda = exp.(0.25 .+ 0.4 .* x .+ b[g] .* x)
    y = map(lambda) do value
        draw = 0
        while draw == 0
            draw = rand(rng, Distributions.Poisson(value))
        end
        Float64(draw)
    end
    return (; y, x, z, g), alpha
end

@testset "zero-truncated Poisson independent slope SD regression" begin
    data, alpha = _truncated_poisson_sd_slope_fixture([-0.55, 0.5])
    formula = bf(@formula(y ~ x + (0 + x | g)),
                 @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, TruncatedPoisson(); data, nAGQ = 15)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sd]
    @test only(nms for (param, nms) in fit.coefnames if param === :sd) ==
          ["(Intercept)", "z"]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.65
    @test length(fit.scales[:sd_group]) == 48
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    pred = predict_parameters(fit, data; se = true)[:sd]
    @test length(pred.value) == length(data.y)
    @test all(isfinite, pred.se)
    simulator = DRModels._marginal_simulator(fit, data)
    @test simulator !== nothing
    ysim = simulator(MersenneTwister(20261031))
    @test length(ysim) == length(data.y)
    @test all(yi -> yi >= 1 && isinteger(yi), ysim)
    boot = bootstrap_result(fit; data, B = 1,
        rng = MersenneTwister(20261101), failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 15
end

@testset "zero-truncated Poisson slope SD likelihood matches scalar integral" begin
    data, _ = _truncated_poisson_sd_slope_fixture([-0.5, 0.3]; G = 8, m = 8,
                                                    seed = 20261029)
    formula = bf(@formula(y ~ x + (0 + x | g)),
                 @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, TruncatedPoisson(); data, nAGQ = 31, se = false)
    theta = [0.25, 0.4, -0.5, 0.3]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:8
        rows = findall(==(gi), data.g)
        integrand = u -> begin
            tau = exp(theta[3] + theta[4] * data.z[first(rows)])
            ll = sum(rows) do i
                eta = theta[1] + theta[2] * data.x[i] + tau * data.x[i] * u
                lambda = exp(eta)
                d = Distributions.Poisson(lambda)
                Distributions.logpdf(d, round(Int, data.y[i])) -
                    log1p(-Distributions.pdf(d, 0))
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-10)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 2e-4)
end

@testset "zero-truncated Poisson slope SD regression validates its target" begin
    data, _ = _truncated_poisson_sd_slope_fixture([-0.5, 0.3]; G = 12, m = 8,
                                                    seed = 20261030)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | g)),
        @formula(sd(g) ~ z)), TruncatedPoisson(); data)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | g)),
        @formula(sd(g, coef(intercept)) ~ z)), TruncatedPoisson(); data)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | g)),
        @formula(sd(g, coef(y)) ~ z)), TruncatedPoisson(); data)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | g)),
        @formula(sd(g, coef(x)) ~ z)), TruncatedPoisson();
        data = merge(data, (; x = fill(0.0, length(data.x)))))
end
