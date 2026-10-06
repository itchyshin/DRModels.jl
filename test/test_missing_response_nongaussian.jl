using DRModels
using Test, Random
import Distributions

function _test_missing_column(y, miss)
    ym = Vector{Union{Missing,Float64}}(Float64.(y))
    ym[miss] .= missing
    return ym
end

function _test_subset_rows(data, rows)
    names = Tuple(keys(data))
    vals = map(names) do k
        collect(getproperty(data, k))[rows]
    end
    return NamedTuple{names}(Tuple(vals))
end

function _test_response_missing_parity(form, fam, data_missing, keep;
        fit_kwargs = (;), coef_atol = 1e-7, loglik_atol = 1e-7)
    fit_missing = drm(form, fam; data = data_missing, fit_kwargs...)
    data_observed = _test_subset_rows(data_missing, keep)
    fit_observed = drm(form, fam; data = data_observed, fit_kwargs...)
    miss = findall(.!keep)

    @test nobs(fit_missing) == nobs(fit_observed)
    @test nobs(fit_missing) == count(keep)
    @test coef(fit_missing) ≈ coef(fit_observed) atol = coef_atol rtol = coef_atol
    @test loglik(fit_missing) ≈ loglik(fit_observed) atol = loglik_atol rtol = loglik_atol
    @test length(fitted(fit_missing)) == length(keep)
    @test fitted(fit_missing)[keep] ≈ fitted(fit_observed) atol = coef_atol rtol = coef_atol
    @test all(isnan, residuals(fit_missing)[miss])

    scales = sigma(fit_missing)
    if scales isa AbstractDict
        @test all(
            k === :ordinal_cuts || length(v) == length(keep)
            for (k, v) in pairs(scales)
        )
    else
        @test length(scales) == length(keep)
    end
    return fit_missing, fit_observed
end

@testset "Non-Gaussian response-missing rows" begin
    Random.seed!(20260610)
    n = 90
    x = collect(range(-1.3, 1.4; length = n))
    miss = [4, 17, 58, 81]
    keep = trues(n)
    keep[miss] .= false

    @testset "Student-t" begin
        y = 0.5 .- 0.4 .* x .+ 0.7 .* rand(Distributions.TDist(5.0), n)
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
            Student(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Poisson" begin
        μ = exp.(0.2 .+ 0.35 .* x)
        y = [rand(Distributions.Poisson(μi)) for μi in μ]
        dat = (; y = _test_missing_column(y, miss), x)
        form = bf(@formula(y ~ x))
        fit_missing, fit_observed = _test_response_missing_parity(form, Poisson(), dat, keep)

        bridged = drm_bridge(;
            formula = "y ~ x",
            family = "poisson",
            data = dat,
        )
        @test bridged["coefficients"] ≈ coef(fit_observed)
        @test bridged["loglik"] ≈ loglik(fit_observed)
        @test bridged["nobs"] == nobs(fit_observed)
        @test length(bridged["fitted"]) == n
        @test all(isnan, bridged["residuals"][miss])
        @test bridged["coefficients"] ≈ coef(fit_missing)
    end

    @testset "Poisson random intercept" begin
        g = repeat(1:9, inner = 10)
        μ = exp.(0.1 .+ 0.25 .* x .+ 0.2 .* sin.(g))
        y = [rand(Distributions.Poisson(μi)) for μi in μ]
        dat = (; y = _test_missing_column(y, miss), x, g)
        _test_response_missing_parity(
            bf(@formula(y ~ x + (1 | g))),
            Poisson(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Negative binomial" begin
        θ = 2.8
        μ = exp.(0.3 .+ 0.4 .* x)
        y = [rand(Distributions.NegativeBinomial(θ, θ / (θ + μi))) for μi in μ]
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x), @formula(sigma ~ 1)),
            NegBinomial2(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Truncated negative binomial" begin
        rtnb(r, p) = (while true; k = rand(Distributions.NegativeBinomial(r, p)); k > 0 && return k; end)
        θ = 3.0
        μ = exp.(0.6 .+ 0.25 .* x)
        y = [rtnb(θ, θ / (θ + μi)) for μi in μ]
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x), @formula(sigma ~ 1)),
            TruncatedNegBinomial2(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Beta" begin
        φ = 14.0
        μ = 1 ./ (1 .+ exp.(-(0.1 .+ 0.65 .* x)))
        y = [rand(Distributions.Beta(μi * φ, (1 - μi) * φ)) for μi in μ]
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x), @formula(sigma ~ 1)),
            Beta(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Gamma" begin
        α = 7.0
        μ = exp.(0.4 .+ 0.3 .* x)
        y = [rand(Distributions.Gamma(α, μi / α)) for μi in μ]
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x), @formula(sigma ~ 1)),
            Gamma(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "LogNormal" begin
        y = exp.(0.2 .+ 0.25 .* x .+ 0.35 .* randn(n))
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x), @formula(sigma ~ 1)),
            LogNormal(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Binomial Bernoulli" begin
        p = 1 ./ (1 .+ exp.(-(-0.2 .+ 0.7 .* x)))
        y = [rand() < pi ? 1.0 : 0.0 for pi in p]
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x)),
            Binomial(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Binomial cbind" begin
        ntr = fill(12, n)
        p = 1 ./ (1 .+ exp.(-(0.15 .+ 0.5 .* x)))
        s = [rand(Distributions.Binomial(ntr[i], p[i])) for i in 1:n]
        fail = ntr .- s
        miss_fail = [7, 33]
        keep2 = trues(n)
        keep2[vcat(miss, miss_fail)] .= false
        dat = (;
            s = _test_missing_column(s, miss),
            fail = _test_missing_column(fail, miss_fail),
            x,
        )
        _test_response_missing_parity(
            bf(@formula(cbind(s, fail) ~ x)),
            Binomial(),
            dat,
            keep2,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Beta-binomial" begin
        ntr = fill(15, n)
        φ = 10.0
        p = 1 ./ (1 .+ exp.(-(0.2 .+ 0.45 .* x)))
        s = [rand(Distributions.BetaBinomial(ntr[i], p[i] * φ, (1 - p[i]) * φ)) for i in 1:n]
        fail = ntr .- s
        miss_fail = [6, 39]
        keep2 = trues(n)
        keep2[vcat(miss, miss_fail)] .= false
        dat = (;
            s = _test_missing_column(s, miss),
            fail = _test_missing_column(fail, miss_fail),
            x,
        )
        _test_response_missing_parity(
            bf(@formula(cbind(s, fail) ~ x), @formula(sigma ~ 1)),
            BetaBinomial(),
            dat,
            keep2,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Zero-one beta" begin
        φ = 10.0
        μ = 1 ./ (1 .+ exp.(-(0.25 .+ 0.55 .* x)))
        y = Vector{Float64}(undef, n)
        for i in 1:n
            if rand() < 0.2
                y[i] = rand() < 0.45 ? 1.0 : 0.0
            else
                y[i] = rand(Distributions.Beta(μ[i] * φ, (1 - μ[i]) * φ))
            end
        end
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
            ZeroOneBeta(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Tweedie" begin
        function rtweedie(μ, φ, p)
            λ = μ^(2 - p) / (φ * (2 - p))
            γ = φ * (p - 1) * μ^(p - 1)
            sh = (2 - p) / (p - 1)
            N = rand(Distributions.Poisson(λ))
            return N == 0 ? 0.0 : rand(Distributions.Gamma(N * sh, γ))
        end
        μ = exp.(0.4 .+ 0.2 .* x)
        y = [rtweedie(μi, 1.6, 1.45) for μi in μ]
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)),
            Tweedie(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end

    @testset "Cumulative logit" begin
        θ = [-0.8, 0.2, 1.0]
        η = 0.65 .* x
        y = Vector{Float64}(undef, n)
        for i in 1:n
            u = rand()
            yi = 4
            for k in 1:3
                if u < 1 / (1 + exp(-(θ[k] - η[i])))
                    yi = k
                    break
                end
            end
            y[i] = yi
        end
        dat = (; y = _test_missing_column(y, miss), x)
        _test_response_missing_parity(
            bf(@formula(y ~ x)),
            CumulativeLogit(),
            dat,
            keep,
            coef_atol = 1e-6,
            loglik_atol = 1e-6,
        )
    end
end
