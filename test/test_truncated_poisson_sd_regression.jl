# Predictor-dependent latent intercept SDs for zero-truncated Poisson.
using DRModels, Test, Random, QuadGK
import Distributions

_rtpois_sd_tp(rng, lambda) = (while true
    value = rand(rng, Distributions.Poisson(lambda))
    value > 0 && return value
end)

function _truncated_poisson_sd_fixture(alpha; G = 40, m = 30,
                                        seed = 20261022)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, G * m)
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    lambda = exp.(0.55 .+ 0.25 .* x .+ b[g])
    y = Float64.([_rtpois_sd_tp(rng, value) for value in lambda])
    return (; y, x, z, g), alpha
end

@testset "zero-truncated Poisson predictor-dependent intercept SD" begin
    data, alpha = _truncated_poisson_sd_fixture([-0.65, 0.5])
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z))
    fit = drm(formula, TruncatedPoisson(); data, nAGQ = 15)
    @test fit.converged
    @test first.(fit.blocks) == [:mu, :sd]
    @test only(nms for (param, nms) in fit.coefnames if param === :sd) ==
          ["(Intercept)", "z"]
    @test sign(coef(fit, :sd)[2]) == sign(alpha[2])
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.6
    @test length(fit.scales[:sd_group]) == 40
    @test predict_parameters(fit, data)[:sd] ≈ fit.scales[:sd]
    pred = predict_parameters(fit, data; se = true)[:sd]
    @test length(pred.value) == length(data.y)
    @test all(isfinite, pred.se)
    simulator = DRModels._marginal_simulator(fit, data)
    @test simulator !== nothing
    ysim = simulator(MersenneTwister(20261024))
    @test length(ysim) == length(data.y)
    @test all(yi -> yi >= 1 && isinteger(yi), ysim)
    boot = bootstrap_result(fit; data, B = 1,
        rng = MersenneTwister(20261026), failures = :skip)
    @test boot.attempted == 1
    @test boot.used == 1
    @test boot.failed == 0
    @test DRModels._bootstrap_refit_kwargs(fit).nAGQ == 15
end

@testset "zero-truncated Poisson SD likelihood matches scalar integral" begin
    data, _ = _truncated_poisson_sd_fixture([-0.5, 0.3]; G = 8, m = 8,
                                             seed = 20261023)
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z))
    fit = drm(formula, TruncatedPoisson(); data, nAGQ = 31, se = false)
    theta = [0.55, 0.25, -0.5, 0.3]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:8
        rows = findall(==(gi), data.g)
        integrand = u -> begin
            b = exp(theta[3] + theta[4] * data.z[first(rows)]) * u
            ll = sum(rows) do i
                eta = theta[1] + theta[2] * data.x[i] + b
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

@testset "zero-truncated Poisson SD regression validates its boundary" begin
    data, _ = _truncated_poisson_sd_fixture([-0.5, 0.3]; G = 12, m = 8,
                                             seed = 20261025)
    route(formula; kwargs...) = drm(formula, TruncatedPoisson(); data, kwargs...)
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 + x | g)),
        @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sd(h) ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sd(g) ~ z), @formula(hu ~ 1)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sd(g) ~ z), @formula(sigma ~ z)))
    @test_throws ArgumentError route(bf(@formula(y ~ x + (1 | g)),
        @formula(sd(g) ~ z)); nAGQ = 0)
    varying = merge(data, (; z = copy(data.z)))
    varying.z[1] = -varying.z[1]
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g)),
        @formula(sd(g) ~ z)), TruncatedPoisson(); data = varying)
end

@testset "zero-truncated Poisson SD regression follows observed-row response policy" begin
    data, _ = _truncated_poisson_sd_fixture([-0.55, 0.4]; G = 16, m = 8,
                                             seed = 20261027)
    formula = bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z))
    observed_rows = setdiff(eachindex(data.y), [3])
    clean = drm(formula, TruncatedPoisson();
                data = DRModels._subset_table_rows(data, observed_rows), nAGQ = 19)
    for bad_response in (missing, NaN)
        y_missing = Vector{Union{Missing,Float64}}(data.y)
        y_missing[3] = bad_response
        incomplete = merge(data, (; y = y_missing))
        io = IOBuffer()
        fit = Base.CoreLogging.with_logger(
            Base.CoreLogging.SimpleLogger(io, Base.CoreLogging.Warn)) do
            drm(formula, TruncatedPoisson(); data = incomplete, nAGQ = 19)
        end
        @test occursin("dropped", String(take!(io)))
        @test nobs(fit) == nobs(clean)
        @test loglik(fit) ≈ loglik(clean) atol = 1e-8
        @test coef(fit, :sd) ≈ coef(clean, :sd) atol = 1e-7
        @test isnan(fit.obs[:mu][3])
        @test fit.scales[:aghq_nodes] == clean.scales[:aghq_nodes] == [19.0]
        @test fit.scales[:sd_group] ≈ clean.scales[:sd_group]
        @test fit.scales[:sd_group_index] == clean.scales[:sd_group_index]
        @test length(fit.scales[:sd]) == length(data.y)
        @test fit.scales[:sd] ≈ predict_parameters(fit, incomplete)[:sd]
    end
end

@testset "zero-truncated Poisson SD regression: a missing response fits with a fresh formula object" begin
    # The fit used to rely on a schema left in the formula object by an earlier fit: a fresh object
    # failed with `KeyError: key 0.0` when the response column was typed `Union{Missing,Float64}`.
    data, _ = _truncated_poisson_sd_fixture([-0.55, 0.4]; G = 16, m = 8, seed = 20261027)
    y_missing = Vector{Union{Missing,Float64}}(data.y); y_missing[3] = missing
    incomplete = merge(data, (; y = y_missing))
    fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)), TruncatedPoisson();
              data = incomplete, nAGQ = 19)
    @test isnan(fit.obs[:mu][3])
    @test length(fit.scales[:sd]) == length(data.y)
    @test fit.scales[:sd] ≈ predict_parameters(fit, incomplete)[:sd]
end
