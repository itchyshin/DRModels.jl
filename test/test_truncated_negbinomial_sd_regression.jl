# Predictor-dependent random-intercept SD for zero-truncated NB2.
using DRModels, Test, Random, QuadGK
import Distributions

function _truncated_nb2_sd_fixture(alpha; G = 48, m = 24, seed = 20261021)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = exp.(0.4 .+ 0.2 .* x .+ b[g])
    sigma = 0.35
    size = 1 / sigma^2
    y = map(mu) do value
        draw = 0
        while draw == 0
            draw = rand(rng, Distributions.NegativeBinomial(size, size / (size + value)))
        end
        draw
    end
    return (; y = Float64.(y), x, z, g), alpha
end

function _truncated_nb2_slope_sd_fixture(alpha; G = 56, m = 24,
                                          seed = 20261027)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = collect(range(-1.0, 1.0; length = G))
    z = zgroup[g]
    x = randn(rng, length(g))
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    mu = exp.(0.25 .+ 0.4 .* x .+ b[g] .* x)
    sigma = 0.5
    size = 1 / sigma^2
    y = map(mu) do value
        draw = 0
        while draw == 0
            draw = rand(rng, Distributions.NegativeBinomial(size,
                size / (size + value)))
        end
        draw
    end
    return (; y = Float64.(y), x, z, g), alpha
end

@testset "zero-truncated NB2 independent slope SD regression" begin
    data, alpha = _truncated_nb2_slope_sd_fixture([-0.55, 0.55])
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, TruncatedNegBinomial2(); data)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test only(nms for (param, nms) in fit.coefnames if param === :sd) ==
          ["(Intercept)", "z"]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    @test length(fit.scales[:sd_group]) == 56
end

@testset "zero-truncated NB2 slope SD likelihood matches scalar integral" begin
    data, _ = _truncated_nb2_slope_sd_fixture([-0.5, 0.3]; G = 8, m = 8,
                                                seed = 20261028)
    formula = bf(@formula(y ~ x + (0 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, coef(x)) ~ z))
    fit = drm(formula, TruncatedNegBinomial2(); data, nAGQ = 31, se = false)
    theta = [0.25, 0.4, log(0.5), -0.5, 0.3]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:8
        rows = findall(==(gi), data.g)
        integrand = u -> begin
            tau = exp(theta[4] + theta[5] * data.z[first(rows)])
            ll = sum(rows) do i
                mu = exp(theta[1] + theta[2] * data.x[i] +
                         tau * data.x[i] * u)
                size = exp(-2 * theta[3])
                d = Distributions.NegativeBinomial(size, size / (size + mu))
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

@testset "zero-truncated NB2 latent SD regression" begin
    data, alpha = _truncated_nb2_sd_fixture([-0.6, 0.4])
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, coef(intercept)) ~ z))
    fit = drm(formula, TruncatedNegBinomial2(); data)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sigma, :sd]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.65
    @test length(fit.scales[:sd_group]) == 48
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
end

@testset "zero-truncated NB2 SD regression recovers a negative predictor effect" begin
    data, alpha = _truncated_nb2_sd_fixture([-0.6, -0.4]; seed = 202610215)
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ z))
    fit = drm(formula, TruncatedNegBinomial2(); data)
    @test fit.converged
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.65
end

@testset "zero-truncated NB2 SD regression refuses unsupported terms" begin
    data, _ = _truncated_nb2_sd_fixture([-0.5, 0.3]; G = 12, m = 8,
                                        seed = 202610216)
    route(formula; kwargs...) = drm(formula, TruncatedNegBinomial2(); data, kwargs...)
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 + x | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)))
    # A keyed `hu ~ ...` is not refused: the route hands it to the shared
    # NegBinomial2 hurdle fitter (src/negbinomial.jl, `haskey(rhs, :hu)`).
    varying = merge(data, (; z = copy(data.z)))
    varying.z[1] = -varying.z[1]
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)), TruncatedNegBinomial2();
        data = varying)
    incomplete = merge(data, (; y = Union{Missing,Float64}[data.y[1:end-1]...;
                                                   missing]))
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)),
        @formula(sigma ~ 1), @formula(sd(g) ~ z)), TruncatedNegBinomial2();
        data = incomplete)
    fixed = bf(@formula(y ~ x), @formula(sigma ~ 1))
    @test_throws ArgumentError drm(fixed, TruncatedNegBinomial2(); data,
                                   nAGQ = 5)
end

@testset "zero-truncated NB2 SD preserves quadrature in bootstrap refits" begin
    data, _ = _truncated_nb2_sd_fixture([-0.55, 0.3]; G = 24, m = 12,
                                        seed = 202610212)
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ z))
    fit = drm(formula, TruncatedNegBinomial2(); data, nAGQ = 3)
    @test only(fit.scales[:aghq_nodes]) == 3
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 3
    boot = bootstrap_result(fit; data, B = 1, rng = MersenneTwister(202610213),
                            failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
end

@testset "zero-truncated NB2 SD likelihood matches scalar integral" begin
    data, _ = _truncated_nb2_sd_fixture([-0.5, 0.3]; G = 8, m = 6,
                                        seed = 202610214)
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g) ~ z))
    fit = drm(formula, TruncatedNegBinomial2(); data, nAGQ = 31, se = false)
    theta = [0.4, 0.2, log(0.35), -0.5, 0.3]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:8
        rows = findall(==(gi), data.g)
        integrand = u -> begin
            ll = sum(rows) do i
                tau = exp(theta[4] + theta[5] * data.z[i])
                mu = exp(theta[1] + theta[2] * data.x[i] + tau * u)
                size = exp(-2 * theta[3])
                d = Distributions.NegativeBinomial(size, size / (size + mu))
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
