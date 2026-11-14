# Bivariate Gaussian with `(1 | p | g)` in mu1 and mu2 accepts rows with one response missing under
# `missing = miss_control(response = "include")`, as drmTMB 0.8.1 does. Such a row contributes the
# marginal likelihood of the response that was observed (its unobserved coordinate is dropped from
# the group's covariance), the same rule as the fixed-effect bivariate route.
#
# Before, this route refused: "needs both responses observed on every row".
#
# Reference: test/fixtures/biv_re_include/include_reference.csv, written by include_r.R in the same
# folder (drmTMB 0.8.1, R 4.6.1, Totoro) from test/fixtures/fitted_modes_bivariate/data.csv.
# Bounds: 1e-4 on logLik, 1e-6 on fitted values (those of test_fitted_modes_bivariate.jl); not loosened.
using DRModels
using Test
using DelimitedFiles
using ForwardDiff
using LinearAlgebra

function _bi_data()
    m, h = readdlm(joinpath(@__DIR__, "fixtures", "fitted_modes_bivariate", "data.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = n == "g" ? String.(c) :
            any(x -> x == "NA", c) ? Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] : Float64.(c)
        push!(cols, Symbol(n) => v)
    end
    return (; cols...)
end

@testset "bivariate (1 | p | g): rows with one response missing, miss_control(response = include)" begin
    d = _bi_data()
    m = readdlm(joinpath(@__DIR__, "fixtures", "biv_re_include", "include_reference.csv"), ',', Any; skipstart = 1)
    r1 = Float64.(m[:, 2]); r2 = Float64.(m[:, 3]); rll = Float64(m[1, 4]); rnobs = Int(m[1, 5])
    f = bf(mu1 = @formula(y1m ~ x1 + (1 | p | g)), mu2 = @formula(y2m ~ x2 + (1 | p | g)),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    inc = miss_control(response = "include")
    fit = drm(f, Gaussian(); data = d, missing = inc)

    @testset "against drmTMB" begin
        @test fit.converged
        @test nobs(fit) == rnobs                                  # rows with at least one response
        @test abs(loglikelihood(fit) - rll) < 1e-4
        c = fitted(fit)
        @test length(c[:mu1]) == length(c[:mu2]) == length(r1)
        @test maximum(abs.(c[:mu1] .- r1)) < 1e-6
        @test maximum(abs.(c[:mu2] .- r2)) < 1e-6
    end

    @testset "the likelihood is the marginal of the observed cells (independent dense Gaussian)" begin
        b1 = coef(fit, :mu1); b2 = coef(fit, :mu2)
        s1 = exp(only(coef(fit, :sigma1))); s2 = exp(only(coef(fit, :sigma2)))
        rho = DRModels.RHO_GUARD * tanh(only(coef(fit, :rho12)))
        Lam = Matrix(fit.ranef.Sigma_g)
        ll = 0.0
        for g in unique(d.g)
            rows = findall(==(g), d.g)
            idx = Tuple{Int,Int}[]                                # (row, response) of every observed cell
            for i in rows
                ismissing(d.y1m[i]) || push!(idx, (i, 1))
                ismissing(d.y2m[i]) || push!(idx, (i, 2))
            end
            isempty(idx) && continue
            y = [k == 1 ? d.y1m[i] - (b1[1] + b1[2] * d.x1[i]) : d.y2m[i] - (b2[1] + b2[2] * d.x2[i]) for (i, k) in idx]
            V = zeros(length(idx), length(idx))
            for (a, (i, k)) in enumerate(idx), (c, (j, l)) in enumerate(idx)
                V[a, c] = Lam[k, l] + (i == j ? (k == l ? (k == 1 ? s1^2 : s2^2) : rho * s1 * s2) : 0.0)
            end
            ll += -0.5 * (length(y) * log(2π) + logdet(Symmetric(V)) + dot(y, Symmetric(V) \ y))
        end
        @test abs(ll - loglikelihood(fit)) < 1e-8
    end

    @testset "gradient: ForwardDiff against central finite differences, and zero at the optimum" begin
        θ = copy(fit.theta) .+ 0.05 .* cos.(1:length(fit.theta))   # a point off the optimum
        g = ForwardDiff.gradient(fit.nll, θ)
        fd = [(h = 1e-6; e = zeros(length(θ)); e[k] = h; (fit.nll(θ .+ e) - fit.nll(θ .- e)) / (2h)) for k in eachindex(θ)]
        @test maximum(abs.(g .- fd) ./ (1 .+ abs.(fd))) < 1e-6
        @test maximum(abs.(ForwardDiff.gradient(fit.nll, fit.theta))) < 1e-3
    end

    @testset "complete rows are unchanged, and a missing cell never enters" begin
        drop = drm(f, Gaussian(); data = d)                       # default: rows with a missing response are dropped
        @test nobs(drop) < nobs(fit)
        @test abs(loglikelihood(drop) - loglikelihood(fit)) > 1                # the partial rows carry information
        fc = drm(bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                    sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1)),
                 Gaussian(); data = d, missing = inc)
        fc2 = drm(bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
                     sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1)),
                  Gaussian(); data = d)
        @test fc.theta == fc2.theta && fc.loglik == fc2.loglik     # complete data: include = drop, bit for bit
    end
end
