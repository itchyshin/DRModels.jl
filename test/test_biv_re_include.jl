# Bivariate Gaussian with `(1 | p | g)` in mu1 and mu2 accepts rows with one response missing under
# `missing = miss_control(response = "include")`, as drmTMB 0.8.1 does. Such a row contributes the
# marginal likelihood of the response that was observed (its unobserved coordinate is dropped from
# the group's covariance), the same rule as the fixed-effect bivariate route.
#
# Before, this route refused: "needs both responses observed on every row".
#
# Reference: test/fixtures/biv_re_include/include_reference.csv, written by include_r.R in the same
# folder (drmTMB 0.8.1, R 4.6.1) from test/fixtures/fitted_modes_bivariate/data.csv.
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

# `method = :REML` together with `miss_control(response = "include")` is accepted on this route.
# drmTMB 0.8.1 refuses that combination ("REML is not implemented with explicit missing-data
# engines yet"). Whether this package should refuse it too is a maintainer decision. This test
# only pins what the route returns today: the REML log-likelihood of the observed cells, written
# here as a dense computation from the model (ML log-likelihood at the generalised least squares
# coefficients, minus 0.5 logdet(X' V^-1 X), plus 0.5 p log(2 pi)).
@testset "bivariate (1 | p | g): REML with include equals a dense REML likelihood of the observed cells" begin
    d = _bi_data()
    f = bf(mu1 = @formula(y1m ~ x1 + (1 | p | g)), mu2 = @formula(y2m ~ x2 + (1 | p | g)),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    fit = drm(f, Gaussian(); data = d, missing = miss_control(response = "include"), method = :REML)
    @test fit.converged
    s1 = exp(only(coef(fit, :sigma1))); s2 = exp(only(coef(fit, :sigma2)))
    rho = DRModels.RHO_GUARD * tanh(only(coef(fit, :rho12)))
    Lam = Matrix(fit.ranef.Sigma_g)
    p = 4
    XtViX = zeros(p, p); XtViy = zeros(p); yViy = 0.0; logdetV = 0.0; N = 0
    for g in unique(d.g)
        rows = findall(==(g), d.g)
        idx = Tuple{Int,Int}[]                                    # (row, response) of every observed cell
        for i in rows
            ismissing(d.y1m[i]) || push!(idx, (i, 1))
            ismissing(d.y2m[i]) || push!(idx, (i, 2))
        end
        isempty(idx) && continue
        y = [k == 1 ? d.y1m[i] : d.y2m[i] for (i, k) in idx]
        X = [k == 1 ? (c == 1 ? 1.0 : c == 2 ? d.x1[i] : 0.0) : (c == 3 ? 1.0 : c == 4 ? d.x2[i] : 0.0)
             for (i, k) in idx, c in 1:p]
        V = zeros(length(idx), length(idx))
        for (a, (i, k)) in enumerate(idx), (c, (j, l)) in enumerate(idx)
            V[a, c] = Lam[k, l] + (i == j ? (k == l ? (k == 1 ? s1^2 : s2^2) : rho * s1 * s2) : 0.0)
        end
        Vs = Symmetric(V)
        XtViX .+= X' * (Vs \ X); XtViy .+= X' * (Vs \ y); yViy += dot(y, Vs \ y)
        logdetV += logdet(Vs); N += length(y)
    end
    beta = XtViX \ XtViy
    quad = yViy - dot(XtViy, beta)
    ll_ml = -0.5 * (N * log(2π) + logdetV + quad)
    ll_reml = ll_ml - 0.5 * logdet(Symmetric(XtViX)) + 0.5 * p * log(2π)
    @test abs(ll_reml - loglikelihood(fit)) < 1e-8
    @test maximum(abs.(vcat(coef(fit, :mu1), coef(fit, :mu2)) .- beta)) < 1e-6   # fixed effects are the GLS values
end

# REML needs a full-rank fixed-effect design among the OBSERVED rows. With `miss_control(response =
# "include")` a covariate can be constant, or equal to another covariate, on the rows where one response is
# observed although it varies in the full data. The restricted likelihood has logdet(X' V^-1 X) = -Inf there;
# before, the fit stopped with a raw `SingularException: ... Zero pivot found at index 3`. It now refuses by
# name. ML is not affected (it keeps its own singular-Hessian warning).
@testset "bivariate (1 | p | g): REML refuses a rank-deficient design of the observed rows" begin
    d0 = _bi_data()
    miss1 = ismissing.(d0.y1m)
    @test any(miss1) && !all(miss1)
    zero_on_obs = Float64.(miss1)                                       # 0 on every row where y1m is observed
    alias_on_obs = ifelse.(miss1, reverse(d0.x1), d0.x1)                # equals x1 on every row where y1m is observed
    @test length(unique(zero_on_obs)) == 2 && length(unique(alias_on_obs)) > 1   # varies in the full data
    inc = miss_control(response = "include")
    mu1_zobs = @formula(y1m ~ x1 + zobs + (1 | p | g))
    mu1_aobs = @formula(y1m ~ x1 + aobs + (1 | p | g))
    for (nm, col, mu1) in ((:zobs, zero_on_obs, mu1_zobs), (:aobs, alias_on_obs, mu1_aobs))
        d = merge(d0, NamedTuple{(nm,)}((col,)))
        f = bf(mu1 = mu1, mu2 = @formula(y2m ~ x2 + (1 | p | g)),
               sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
        err = try drm(f, Gaussian(); data = d, missing = inc, method = :REML); nothing catch e e end
        @test err isa ArgumentError
        msg = err isa ArgumentError ? err.msg : sprint(showerror, err)
        @test occursin("full-rank fixed-effect design among the observed rows", msg)
        @test occursin("`mu1` has rank 2 for 3 columns", msg)
        @test occursin(String(nm), msg)
        @test occursin("method = :ML", msg)
    end
end
