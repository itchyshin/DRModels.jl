# simulate(fit; data) for two ordinary random intercepts: draws checked against the
# fitted model's own implied marginal moments, family by family (#1319 follow-up).
#
# Why: the first #1319 tests compared draws with drmTMB output for three families only.
# A Binomial two-intercept fit stored no `trials`, so `simulate` threw "does not carry
# `trials`" (also for `re = :zero`). This file checks every family the route admits.
#
# Implied moments. For row i the linear predictor is eta_i (fixed part) and the two
# intercepts sum to u ~ N(0, s1^2 + s2^2) (independent fields). Given u the response has
# conditional mean m_i(u) and variance v_i(u) (closed forms per family below). By total
# expectation and variance,  E y_i = E m_i(u),  Var y_i = E v_i(u) + Var m_i(u),  with the
# u-expectations done by QuadGK (no simulation, so independent of the simulator).
# Statistics compared: the mean over rows of E y_i, and the mean over rows of Var y_i.
# Monte Carlo bound (stated, not tuned): with N draws, the row-averaged sample mean has SD
# at most sqrt(avg Var / N) (Cauchy-Schwarz), and the row-averaged sample variance has
# relative SD at most sqrt((kmax - 1) / N) where kmax is the largest row kurtosis of the
# draws. Both are tested at 5 SD.
using DRModels
using Test
using Random
using Statistics
using DelimitedFiles
using QuadGK
using StatsModels: @formula

const _SMM_DIR = joinpath(@__DIR__, "fixtures", "gap_v2_crashes")
const _SMM_FACTORS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
const _SMM = let (m, h) = readdlm(joinpath(_SMM_DIR, "data_v2.csv"), ',', Any; header = true)
    h = String.(vec(h))
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(h)
        c = m[:, j]
        v = if n in _SMM_FACTORS
            any(==("NA"), c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] : String.(c)
        elseif any(==("NA"), c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        elseif all(x -> x isa Integer, c)
            Int.(c)
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    d = (; cols...)
    # successes / failures for the two-column Binomial and BetaBinomial forms
    merge(d, (; succ = Float64.(d.y_pois), failc = Float64.(max.(d.tr .- d.y_pois, 0))))
end

# Synthetic data with strong intercepts (SD 0.9 and 0.7) for the families whose fixture
# fits put the intercept SD near zero, where a moment comparison could not tell a missing
# intercept draw from a present one.
const _SYN = let rng = MersenneTwister(20261111), n = 480
    G1, G2 = 12, 8
    gi = repeat(1:G1, inner = n ÷ G1); hi = repeat(1:G2, outer = n ÷ G2)
    x = randn(rng, n)
    u = 0.9 .* randn(rng, G1); v = 0.7 .* randn(rng, G2)
    η = 0.2 .+ 0.4 .* x .+ u[gi] .+ v[hi]
    ntr = rand(rng, 3:10, n)
    p = @. 1 / (1 + exp(-η))
    ysucc = Float64[sum(rand(rng) < p[i] for _ in 1:ntr[i]) for i in 1:n]
    pb = [rand(rng, DRModels.Distributions.Beta(p[i] * 4, (1 - p[i]) * 4)) for i in 1:n]   # BetaBinomial, phi = 4
    bbsucc = Float64[sum(rand(rng) < pb[i] for _ in 1:ntr[i]) for i in 1:n]
    cuts = [-0.8, 0.4, 1.5]
    yord = Int[DRModels._rand_cumulative_logit(rng, η[i] - 0.2, cuts) for i in 1:n]
    (; g = ["g" * lpad(string(k), 2, '0') for k in gi], g2 = ["k" * string(k) for k in hi], x1 = x,
       y_st = η .+ 0.5 .* rand(rng, DRModels.Distributions.TDist(6), n),
       y_tw = [DRModels._rand_tweedie(rng, exp(0.5 * η[i]), 0.5, 1.4) for i in 1:n],
       y_ord = yord, y_bin = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-(η .+ 1.5)))),
       succ = ysucc, failc = Float64.(ntr) .- ysucc,
       bbs = bbsucc, bbf = Float64.(ntr) .- bbsucc)
end

# --- conditional (mean, variance) of one response given eta (incl. the intercepts) --------
_lg(x) = 1 / (1 + exp(-x))
_cm_gaussian(η, σ, _) = (η, σ^2)
_cm_student(η, σ, ν) = (η, σ^2 * ν / (ν - 2))
_cm_lognormal(η, σ, _) = (exp(η + σ^2 / 2), (exp(σ^2) - 1) * exp(2η + σ^2))
_cm_gamma_prec(η, σ, _) = (exp(η), exp(2η) * σ^2)           # shape = 1 / σ^2
_cm_poisson(η, σ, _) = (exp(η), exp(η))
_cm_nb2(η, σ, _) = (exp(η), exp(η) + exp(2η) * σ^2)         # theta = 1 / σ^2
_cm_beta(η, σ, _) = (p = _lg(η); (p, p * (1 - p) / (1 / σ^2 + 1)))
_cm_tweedie(η, σ, p) = (exp(η), σ^2 * exp(η)^p)

# Marginal (mean, variance) of row i: integrate the conditional moments over u.
function _marginal_moments(cm, η, σ, aux, s)
    s <= 0 && return cm(η, σ, aux)
    w(u) = exp(-u^2 / (2s^2)) / (s * sqrt(2π))
    lim = 9s
    Em = quadgk(u -> cm(η + u, σ, aux)[1] * w(u), -lim, lim; rtol = 1e-9)[1]
    Em2 = quadgk(u -> cm(η + u, σ, aux)[1]^2 * w(u), -lim, lim; rtol = 1e-9)[1]
    Ev = quadgk(u -> cm(η + u, σ, aux)[2] * w(u), -lim, lim; rtol = 1e-9)[1]
    return Em, Ev + Em2 - Em^2
end

function _check_moments(label, fit, cm; N = 4000, seed = 20261111, eta = nothing, aux = nothing, data = _SMM, min_share = 0.05)
    sds = re_sd(fit)
    s = sqrt(sum(abs2, values(sds)))
    η = eta === nothing ? Vector{Float64}(predict(fit, data; type = :link)) : eta
    σ = haskey(fit.scales, :sigma) ? Vector{Float64}(fit.scales[:sigma]) : fill(NaN, length(η))
    aux === nothing && (aux = haskey(fit.scales, :nu) ? Vector{Float64}(fit.scales[:nu]) : fill(NaN, length(η)))
    S = simulate(fit; nsim = N, data = data, rng = MersenneTwister(seed))
    @test size(S) == (length(η), N)
    @test all(isfinite, S)
    mom = [_marginal_moments(cm, η[i], σ[i], aux[i], s) for i in eachindex(η)]
    Ey = mean(first, mom); Vy = mean(last, mom)
    ybar = mean(S); vbar = mean(var(S; dims = 2))
    kmax = maximum(begin
        x = view(S, i, :); m = mean(x); v = var(x)
        v > 0 ? mean((x .- m) .^ 4) / v^2 : 3.0
    end for i in axes(S, 1))
    mean_bound = 5 * sqrt(Vy / N)
    var_bound = 5 * sqrt((kmax - 1) / N) * Vy
    # share of the marginal variance that the two intercepts add (power of the comparison)
    V0 = mean(last(_marginal_moments(cm, η[i], σ[i], aux[i], 0.0)) for i in eachindex(η))
    share = 1 - V0 / Vy
    @info "$label" s Ey ybar Vy vbar mean_bound var_bound share
    @test share >= min_share   # the check has power: the intercepts matter in this fit
    @test abs(ybar - Ey) <= mean_bound
    @test abs(vbar - Vy) <= var_bound
    return S
end

# Binomial / BetaBinomial: trials come from the response columns, so the conditional
# moments need them per row. `trials` is the vector the test derives from the data.
function _check_moments_trials(label, fit, trials, cm_trials; N = 4000, seed = 20261111, data = _SMM)
    sds = re_sd(fit)
    s = sqrt(sum(abs2, values(sds)))
    η = Vector{Float64}(predict(fit, data; type = :link))
    σ = haskey(fit.scales, :sigma) ? Vector{Float64}(fit.scales[:sigma]) : fill(NaN, length(η))
    S = simulate(fit; nsim = N, data = data, rng = MersenneTwister(seed))
    @test size(S) == (length(η), N)
    @test all(isfinite, S)
    @test all(i -> all(0 .<= view(S, i, :) .<= trials[i]), axes(S, 1))
    mom = [_marginal_moments((e, sg, nn) -> cm_trials(e, sg, nn), η[i], σ[i], trials[i], s) for i in eachindex(η)]
    Ey = mean(first, mom); Vy = mean(last, mom)
    ybar = mean(S); vbar = mean(var(S; dims = 2))
    kmax = maximum(begin
        x = view(S, i, :); m = mean(x); v = var(x)
        v > 0 ? mean((x .- m) .^ 4) / v^2 : 3.0
    end for i in axes(S, 1))
    mean_bound = 5 * sqrt(Vy / N)
    var_bound = 5 * sqrt((kmax - 1) / N) * Vy
    V0 = mean(last(_marginal_moments((e, sg, nn) -> cm_trials(e, sg, nn), η[i], σ[i], trials[i], 0.0)) for i in eachindex(η))
    share = 1 - V0 / Vy
    # power: either the intercepts add visible variance, or (Bernoulli, whose variance
    # barely depends on the intercepts) they move the marginal mean by more than the bound
    E0 = mean(first(cm_trials(η[i], σ[i], trials[i])) for i in eachindex(η))
    meanshift = abs(Ey - E0)
    @info "$label" s Ey ybar Vy vbar mean_bound var_bound share meanshift
    @test share >= 0.05 || meanshift > 2 * mean_bound
    @test abs(ybar - Ey) <= mean_bound
    @test abs(vbar - Vy) <= var_bound
    return S
end
_cm_binomial(η, _σ, n) = (p = _lg(η); (n * p, n * p * (1 - p)))
_cm_betabin(η, σ, n) = (p = _lg(η); φ = 1 / σ^2; (n * p, n * p * (1 - p) * (φ + n) / (φ + 1)))

@testset "simulate: two random intercepts, draws match implied moments (all admitted families)" begin
    R2 = "+ (1 | g) + (1 | g2)"
    f(y) = eval(Meta.parse("@formula($y ~ x1 $R2)"))
    s1 = @formula(sigma ~ 1)

    @testset "Gaussian" begin
        _check_moments("Gaussian", drm(bf(f("y_g"), s1), Gaussian(); data = _SMM), _cm_gaussian)
    end
    @testset "Student" begin
        _check_moments("Student", drm(bf(f("y_st"), s1), Student(); data = _SYN), _cm_student; data = _SYN, min_share = 0.05)
    end
    @testset "LogNormal" begin
        _check_moments("LogNormal", drm(bf(f("y_ln"), s1), LogNormal(); data = _SMM), _cm_lognormal)
    end
    @testset "Gamma" begin
        _check_moments("Gamma", drm(bf(f("y_gam"), s1), Gamma(); data = _SMM), _cm_gamma_prec)
    end
    @testset "Beta" begin
        _check_moments("Beta", drm(bf(f("y_beta"), s1), Beta(); data = _SMM), _cm_beta)
    end
    @testset "Poisson" begin
        _check_moments("Poisson", drm(bf(f("y_pois")), Poisson(); data = _SMM), _cm_poisson)
    end
    @testset "NegBinomial2" begin
        _check_moments("NB2", drm(bf(f("y_nb"), s1), NegBinomial2(); data = _SMM), _cm_nb2)
    end
    @testset "Tweedie" begin
        _check_moments("Tweedie", drm(bf(f("y_tw")), Tweedie(); data = _SYN), _cm_tweedie; data = _SYN, min_share = 0.05)
    end
    @testset "CumulativeLogit" begin
        fit = drm(bf(f("y_ord")), CumulativeLogit(); data = _SYN)
        cuts = Vector{Float64}(fit.scales[:ordinal_cuts])
        K = length(cuts) + 1
        # P(y = k | eta) from the cumulative logit; the response is the category index 1..K
        function cm_ord(η, _σ, _)
            cum = [_lg(c - η) for c in cuts]
            p = [cum[1]; [cum[k] - cum[k-1] for k in 2:K-1]; 1 - cum[end]]
            m = sum(k * p[k] for k in 1:K)
            return (m, sum(k^2 * p[k] for k in 1:K) - m^2)
        end
        _check_moments("CumulativeLogit", fit, cm_ord; eta = Vector{Float64}(fit.scales[:ordinal_eta]),
                       data = _SYN, min_share = 0.05)
    end
    @testset "Binomial, 0/1 response (the #1319 'does not carry trials' defect)" begin
        fit = drm(bf(f("y_bin")), Binomial(); data = _SYN)
        _check_moments_trials("Binomial 0/1", fit, ones(fit.nobs), _cm_binomial; data = _SYN)
    end
    @testset "Binomial, successes + failures response" begin
        fit = drm(bf(f("cbind(succ, failc)")), Binomial(); data = _SYN)
        tr = _SYN.succ .+ _SYN.failc
        _check_moments_trials("Binomial cbind", fit, tr, _cm_binomial; data = _SYN)
    end
    @testset "BetaBinomial, successes + failures response" begin
        fit = drm(bf(f("cbind(bbs, bbf)"), s1), BetaBinomial(); data = _SYN)
        tr = _SYN.bbs .+ _SYN.bbf
        _check_moments_trials("BetaBinomial", fit, tr, _cm_betabin; data = _SYN)
    end
    @testset "Binomial crossed fit keeps its trials (Laplace and AGHQ)" begin
        tr = _SYN.succ .+ _SYN.failc
        fl = drm(bf(f("cbind(succ, failc)")), Binomial(); data = _SYN)
        @test haskey(fl.scales, :trials) && fl.scales[:trials] == tr
        fa = drm(bf(f("cbind(succ, failc)")), Binomial(); data = _SYN, marginal = :AGHQ)
        @test haskey(fa.scales, :trials) && fa.scales[:trials] == tr
        S = simulate(fa; nsim = 50, data = _SYN, rng = MersenneTwister(3))
        @test size(S) == (fa.nobs, 50)
        @test all(i -> all(0 .<= view(S, i, :) .<= tr[i]), axes(S, 1))
        # 0/1 response: trials are all ones
        f01 = drm(bf(f("y_bin")), Binomial(); data = _SYN)
        @test f01.scales[:trials] == ones(f01.nobs)
    end
    @testset "a fit from a route that stored no trials: rebuilt from `data`, else refused" begin
        # Mimics a route (for example one from another PR) that leaves `fit.scales` without
        # `:trials`: the simulator rebuilds them from the response columns of `data`.
        strip(fit) = DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov, fit.loglik,
            fit.nobs, fit.converged, fit.means, fit.obs,
            Dict{Symbol,Vector{Float64}}(k => v for (k, v) in fit.scales if k !== :trials),
            fit.formula, fit.nll, fit.nllgrad, fit.ranef, fit.estim_method, fit.reml_loglik,
            fit.ml_loglik, fit.marginal, fit.phylo_penalty, fit.penalty, fit.iterations,
            fit.phylo_scale, fit.uncertainty, fit.geometry)
        fit = strip(drm(bf(f("cbind(succ, failc)")), Binomial(); data = _SYN))
        @test !haskey(fit.scales, :trials)
        tr = _SYN.succ .+ _SYN.failc
        _check_moments_trials("Binomial cbind, trials rebuilt", fit, tr, _cm_binomial; data = _SYN)
        fit01 = strip(drm(bf(f("y_bin")), Binomial(); data = _SYN))
        _check_moments_trials("Binomial 0/1, trials rebuilt", fit01, ones(fit01.nobs), _cm_binomial; data = _SYN)
        # response columns missing from `data`: trials cannot be known, so a clear refusal
        nores = (; g = _SYN.g, g2 = _SYN.g2, x1 = _SYN.x1)
        err = try; simulate(fit; data = nores); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("pass the fitting `data`", sprint(showerror, err))
    end
    @testset "Binomial population-level draw (re = :zero) works too" begin
        # the refusal text offers `re = :zero`; it used to throw "does not carry `trials`"
        fit = drm(bf(f("cbind(succ, failc)")), Binomial(); data = _SYN)
        Z = simulate(fit; nsim = 5, re = :zero, rng = MersenneTwister(2))
        @test size(Z) == (fit.nobs, 5)
        @test all(isfinite, Z)
    end
end
