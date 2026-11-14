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
using StableRNGs

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
# engines yet"), so there is no R reference; the maintainers decided to keep it, with the rank
# refusal below and the missing-at-random and log-likelihood-comparability notes in the docs. The
# check here is an independent dense computation from the model: the REML log-likelihood of the
# observed cells (ML log-likelihood at the generalised least squares coefficients, minus
# 0.5 logdet(X' V^-1 X), plus 0.5 p log(2 pi)).
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

# The ways the observed-row design X_o can lose rank under REML, each refused by name (an ArgumentError
# that names REML, the response, the aliased columns and what to do), never a raw factorisation error:
#   (a) a factor level seen only in rows whose response is missing but kept (include),
#   (b) a covariate constant among the observed rows of mu2 (the mu1 case is the testset above),
#   (c) a factor level whose rows all have weight 0,
#   (d) fewer observed cells than columns (refused earlier, for ML as well).
# A full-rank X_o still fits, and with no missing value `include` equals the default fit under REML.
@testset "bivariate (1 | p | g): REML refusal covers every way the observed-row design loses rank" begin
    d0 = _bi_data()
    n = length(d0.y1m)
    miss1 = ismissing.(d0.y1m); miss2 = ismissing.(d0.y2m)
    inc = miss_control(response = "include")
    mk(mu1, mu2 = @formula(y2m ~ x2 + (1 | p | g))) = bf(mu1 = mu1, mu2 = mu2,
        sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    refused(f, d; kw...) = (try drm(f, Gaussian(); data = d, method = :REML, kw...); nothing catch e e end)

    @testset "(a) a factor level only in rows with a missing response" begin
        fz = [miss1[i] ? "c" : (iseven(i) ? "a" : "b") for i in 1:n]
        d = merge(d0, (; fz))
        f = mk(@formula(y1m ~ x1 + fz + (1 | p | g)))
        err = refused(f, d; missing = inc)
        @test err isa ArgumentError
        msg = err isa ArgumentError ? err.msg : sprint(showerror, err)
        for s in ("method = :REML", "full-rank fixed-effect design among the observed rows", "`mu1` has rank 3 for 4 columns",
                  "fz: c", "rows with a missing `y1m` stay in the fit", "miss_control(response = \"include\")",
                  "method = :ML", "out of `data`")
            @test occursin(s, msg)
        end
        # taking those rows out of `data` removes the level, and the fit works
        keep = findall(!, miss1)
        dsub = NamedTuple{keys(d)}(map(c -> c[keep], values(d)))
        fsub = mk(@formula(y1m ~ x1 + fz + (1 | p | g)))          # a new formula: a formula object caches the levels of its first data
        fit = drm(fsub, Gaussian(); data = dsub, missing = inc, method = :REML)
        @test fit.converged && isfinite(loglikelihood(fit))
        # ML is not refused by this check
        errml = try drm(f, Gaussian(); data = d, method = :ML, missing = inc); nothing catch e e end
        @test errml === nothing || !occursin("full-rank", sprint(showerror, errml))
    end

    @testset "(b) mu2: a covariate constant among the observed rows of y2m" begin
        zz = Float64.(miss2)
        f = mk(@formula(y1m ~ x1 + (1 | p | g)), @formula(y2m ~ x2 + zz + (1 | p | g)))
        err = refused(f, merge(d0, (; zz)); missing = inc)
        @test err isa ArgumentError
        msg = err isa ArgumentError ? err.msg : sprint(showerror, err)
        @test occursin("`mu2` has rank 2 for 3 columns", msg) && occursin("zz", msg) && occursin("method = :ML", msg)
        @test occursin("rows with a missing `y2m` stay in the fit", msg)
    end

    @testset "(c) a level whose rows all have weight 0" begin
        fzw = [iseven(i) ? "a" : "b" for i in 1:n]
        w = [fzw[i] == "b" ? 0.0 : 1.0 for i in 1:n]
        d = merge(d0, (; fz = fzw))
        f = mk(@formula(y1m ~ x1 + fz + (1 | p | g)))
        for kw in ((; weights = w), (; weights = w, missing = inc))
            err = refused(f, d; kw...)
            @test err isa ArgumentError
            msg = err isa ArgumentError ? err.msg : sprint(showerror, err)
            @test occursin("`mu1` has rank 2 for 3 columns", msg) && occursin("fz: b", msg) && occursin("weight 0", msg)
        end
    end

    @testset "(d) fewer observed cells than columns" begin
        y1few = Union{Missing,Float64}[i <= 2 ? d0.y1m[i] : missing for i in 1:n]
        f = mk(@formula(y1few ~ x1 + x2 + (1 | p | g)))
        d = merge(d0, (; y1few))
        err = refused(f, d; missing = inc)
        @test err isa ArgumentError && occursin("fewer than the mu1 coefficients", err.msg) && occursin("y1few", err.msg)
    end

    @testset "full-rank X_o fits, and with no missing value include equals the default under REML" begin
        f = mk(@formula(y1 ~ x1 + (1 | p | g)), @formula(y2 ~ x2 + (1 | p | g)))
        a = drm(f, Gaussian(); data = d0, method = :REML)
        b = drm(f, Gaussian(); data = d0, method = :REML, missing = inc)
        @test a.theta == b.theta && a.loglik == b.loglik && nobs(a) == nobs(b)
        # a fit with kept rows and a full-rank X_o is not refused
        fm = mk(@formula(y1m ~ x1 + (1 | p | g)))
        @test refused(fm, d0; missing = inc) === nothing
    end
end

# The restricted likelihood of the observed cells, derived and checked on generated data (not the fixture):
#   l_R = -(N - p)/2 log(2 pi) - 1/2 log|V| - 1/2 log|X_o' V^-1 X_o| - 1/2 (y - X_o b)' V^-1 (y - X_o b),
# with V the covariance of the N observed cells (group covariance plus the 2 x 2 residual block of each row,
# restricted to the observed coordinates), X_o their stacked design (p columns, full rank) and b the GLS value.
# Three columns in mu1 (a 0/1 covariate), rows missing one response, and rows missing both.
@testset "bivariate (1 | p | g): REML with include on generated data, dense restricted likelihood (p = 5)" begin
    rng = StableRNG(20263)
    n, G = 150, 20
    g = sort!(rand(rng, 1:G, n)); x1 = randn(rng, n); x2 = randn(rng, n); xb = Float64.(rand(rng, n) .< 0.4)
    U = cholesky(Symmetric([0.4 0.1; 0.1 0.8])).L * randn(rng, 2, G)
    e1 = randn(rng, n); e2 = -0.2 .* e1 .+ sqrt(1 - 0.04) .* randn(rng, n)
    y1 = 0.5 .+ 0.2 .* x1 .+ 0.7 .* xb .+ [U[1, i] for i in g] .+ 0.6 .* e1
    y2 = 1.0 .- 0.6 .* x2 .+ [U[2, i] for i in g] .+ 0.9 .* e2
    m1 = rand(rng, n) .< 0.25; m2 = rand(rng, n) .< 0.25
    y1m = Union{Missing,Float64}[m1[i] ? missing : y1[i] for i in 1:n]
    y2m = Union{Missing,Float64}[m2[i] ? missing : y2[i] for i in 1:n]
    @test any(m1 .& m2) && any(xor.(m1, m2))                       # rows missing both, rows missing one
    d = (; g = string.(g), x1, x2, xb, y1m, y2m)
    f = bf(mu1 = @formula(y1m ~ x1 + xb + (1 | p | g)), mu2 = @formula(y2m ~ x2 + (1 | p | g)),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    fit = drm(f, Gaussian(); data = d, missing = miss_control(response = "include"), method = :REML)
    @test fit.converged
    s1 = exp(only(coef(fit, :sigma1))); s2 = exp(only(coef(fit, :sigma2)))
    rho = DRModels.RHO_GUARD * tanh(only(coef(fit, :rho12)))
    Lam = Matrix(fit.ranef.Sigma_g)
    X1 = hcat(ones(n), x1, xb); X2 = hcat(ones(n), x2); p = 5
    XtViX = zeros(p, p); XtViy = zeros(p); yViy = 0.0; ldV = 0.0; N = 0
    for gg in unique(g)
        idx = Tuple{Int,Int}[]
        for i in findall(==(gg), g)
            ismissing(y1m[i]) || push!(idx, (i, 1))
            ismissing(y2m[i]) || push!(idx, (i, 2))
        end
        isempty(idx) && continue
        y = [k == 1 ? y1m[i] : y2m[i] for (i, k) in idx]
        X = [k == 1 ? (c <= 3 ? X1[i, c] : 0.0) : (c > 3 ? X2[i, c-3] : 0.0) for (i, k) in idx, c in 1:p]
        V = zeros(length(idx), length(idx))
        for (a, (i, k)) in enumerate(idx), (b, (j, l)) in enumerate(idx)
            V[a, b] = Lam[k, l] + (i == j ? (k == l ? (k == 1 ? s1^2 : s2^2) : rho * s1 * s2) : 0.0)
        end
        Vs = Symmetric(V)
        XtViX .+= X' * (Vs \ X); XtViy .+= X' * (Vs \ y); yViy += dot(y, Vs \ y); ldV += logdet(Vs); N += length(y)
    end
    b = XtViX \ XtViy
    ll = -0.5 * ((N - p) * log(2π) + ldV + logdet(Symmetric(XtViX)) + (yViy - dot(XtViy, b)))
    @test abs(ll - loglikelihood(fit)) < 1e-8
    @test maximum(abs.(vcat(coef(fit, :mu1), coef(fit, :mu2)) .- b)) < 1e-6
end
