# Focused source test until the Bernoulli dispatcher is wired.  The production
# include will replace this conditional include; it permits the helper to be
# exercised without changing shared dispatcher files owned by other lanes.
using DRModels, Test, Random, QuadGK
import Distributions

let source = joinpath(dirname(@__DIR__), "src", "binomial_sd_regression.jl")
    isfile(source) && !isdefined(DRModels, :_drm_binomial_sd_regression) &&
        Base.include(DRModels, source)
end

function _binomial_sd_fixture(; G = 28, m = 8, trials = 12)
    rng = MersenneTwister(20261017)
    g = repeat(1:G, inner = m)
    zgroup = repeat([0.0, 1.0], inner = G ÷ 2)
    z = zgroup[g]
    x = randn(rng, length(g))
    alpha = [-0.55, 0.55]
    b = exp.(alpha[1] .+ alpha[2] .* zgroup) .* randn(rng, G)
    eta = 0.15 .+ 0.45 .* x .+ b[g]
    prob = 1.0 ./ (1.0 .+ exp.(-eta))
    s = [rand(rng, Distributions.Binomial(trials, p)) for p in prob]
    return (; s = Float64.(s), fail = Float64.(trials .- s), x, z, g), alpha
end

@testset "binomial latent SD regression: q1 fit and reduction" begin
    data, alpha = _binomial_sd_fixture()
    variable = bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sd(g) ~ z))
    fit = DRModels._drm_binomial_sd_regression(variable, Binomial(); data = data)
    @test fit.converged
    @test length(coef(fit, :sd)) == 2
    @test coef(fit, :sd)[2] > 0
    @test abs(coef(fit, :sd)[2] - alpha[2]) < 0.55
    @test all(isfinite, coef(fit))
    @test_throws ArgumentError re_sd(fit)
    @test fit.scales[:sd_group] ≈ exp.(hcat(ones(28), [zeros(14); ones(14)]) * coef(fit, :sd))
    @test Int.(fit.scales[:sd_group_index]) == data.g

    constant = bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sd(g) ~ 1))
    f_sd = DRModels._drm_binomial_sd_regression(constant, Binomial(); data = data)
    f_old = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), Binomial(); data = data)
    @test f_sd.converged && f_old.converged
    @test isapprox(f_sd.loglik, f_old.loglik; atol = 1e-5)
    @test isapprox(coef(f_sd, :mu), coef(f_old, :mu); atol = 1e-4)
    @test isapprox(coef(f_sd, :sd)[1], coef(f_old, :resd)[1]; atol = 1e-4)
end

@testset "Bernoulli response equals one-trial binomial" begin
    counts, _ = _binomial_sd_fixture(; G = 28, m = 8, trials = 1)
    bernoulli = (; y = counts.s, x = counts.x, z = counts.z, g = counts.g)
    bfit = DRModels._drm_binomial_sd_regression(
        bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)), Binomial(); data = bernoulli)
    cfit = DRModels._drm_binomial_sd_regression(
        bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sd(g) ~ z)),
        Binomial(); data = counts)
    @test bfit.converged && cfit.converged
    @test isapprox(bfit.loglik, cfit.loglik; atol = 1e-8)
    @test isapprox(coef(bfit), coef(cfit); atol = 1e-8)
    @test all(0 .< fitted(bfit) .< 1)
end

@testset "binomial latent SD regression: independent fixed-parameter integral" begin
    data, _ = _binomial_sd_fixture(; G = 28, m = 4, trials = 2)
    formula = bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sd(g) ~ z))
    fit5 = DRModels._drm_binomial_sd_regression(formula, Binomial(); data = data, nq = 5)
    fit = DRModels._drm_binomial_sd_regression(formula, Binomial(); data = data, nq = 9)
    fit13 = DRModels._drm_binomial_sd_regression(formula, Binomial(); data = data, nq = 13)
    theta = [0.15, 0.45, -0.55, 0.55]
    normal = Distributions.Normal()
    reference = 0.0
    for gi in 1:28
        rows = findall(==(gi), data.g)
        sdg = exp(theta[3] + theta[4] * data.z[first(rows)])
        integrand = u -> begin
            ll = sum(rows) do i
                p = 1 / (1 + exp(-(theta[1] + theta[2] * data.x[i] + sdg * u)))
                Distributions.logpdf(Distributions.Binomial(2, p), Int(data.s[i]))
            end
            exp(ll) * Distributions.pdf(normal, u)
        end
        value, _ = QuadGK.quadgk(integrand, -10.0, 10.0; rtol = 1e-11)
        reference += log(value)
    end
    @test isapprox(-fit.nll(theta), reference; atol = 1e-5)
    @test abs(fit5.nll(theta) - fit13.nll(theta)) < 2e-3
    @test abs(fit.nll(theta) - fit13.nll(theta)) < 1e-5
    @test fit.scales[:aghq_nodes] == [9.0]
end

@testset "binomial latent SD regression: unsupported cells refuse explicitly" begin
    data, _ = _binomial_sd_fixture(; G = 6, m = 4, trials = 2)
    route(f; kwargs...) = DRModels._drm_binomial_sd_regression(f, Binomial(); data = data, kwargs...)
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                                        @formula(sd(h) ~ z)))
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 + x | g)),
                                        @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                                        @formula(sd(g) ~ x)))
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                                        @formula(sigma ~ x), @formula(sd(g) ~ z)))
    @test_throws ArgumentError route(bf(@formula(cbind(s, fail) ~ x + (1 | g)),
                                        @formula(sd(g) ~ z)); marginal = :Laplace)
end
