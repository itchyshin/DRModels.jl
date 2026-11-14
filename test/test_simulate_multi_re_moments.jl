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
#
# Group structure (review follow-up). The pooled row moments cannot see whether rows that
# share a level of `g` or `g2` share their random effect: drawing one effect per ROW leaves
# every row's marginal moments unchanged. So the file also checks the between-group
# structure, computed from the same conditional moments (no simulation). Rows i and j are
# conditionally independent given the effects, so Cov(y_i, y_j) = Cov(m_i(u_i), m_j(u_j))
# with u_i = a_g(i) + b_h(i); it is nonzero only for rows sharing a `g` level, an `h` level
# or both, and Var(y_i) = E v_i + Var m_i. From the n x n row covariance C and the
# group-mean operators P1 (levels of g) and P2 (levels of g2) three statistics follow:
#   T1 = mean over g levels of Var(group mean),   T2 = the same for g2 levels,
#   T3 = Cov(sum of g-level means, sum of g2-level means).
# Draws give unbiased estimates of each (centred at the exact expected group means), with a
# standard error read off the draws themselves: z_d = per-draw value, SE = sd(z) / sqrt(N),
# which includes the dependence between levels. Each is tested at 5 SE (about 5.7e-7 per
# check under normality). T1 and T2 see a missing, per-row or wrongly scaled draw; T3 sees
# two intercepts that are not independent (one normal vector reused for both).
# Checks: 3 per family case, 14 family cases = 42, plus 14 pooled mean and 14 pooled
# variance checks: about 70 stochastic checks, so under 4e-5 chance that a different seed
# would fail; the seed is fixed, so the file is deterministic.
using DRModels
using Test
using Random
using Statistics
using DelimitedFiles
using QuadGK
using LinearAlgebra: dot, Diagonal, diag
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
    d0 = (; g = ["g" * lpad(string(k), 2, '0') for k in gi], g2 = ["k" * string(k) for k in hi], x1 = x,
       y_st = η .+ 0.5 .* rand(rng, DRModels.Distributions.TDist(6), n),
       y_tw = [DRModels._rand_tweedie(rng, exp(0.5 * η[i]), 0.5, 1.4) for i in 1:n],
       y_ord = yord, y_bin = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-(η .+ 1.5)))),
       succ = ysucc, failc = Float64.(ntr) .- ysucc,
       bbs = bbsucc, bbf = Float64.(ntr) .- bbsucc)
    # Gamma (shape 4), Beta (precision 5) and NB2 (size 3) responses, drawn after everything
    # above so that the columns above do not change. Gamma and NB2 use the mean exp(0.6 eta)
    # (intercept SDs about 0.54 and 0.42): at exp(eta) the draws are so heavy-tailed that the
    # group-structure checks lose their power. The fixture fits of these three put one
    # of the two intercept SDs near zero, where two intercepts that are not independent
    # (one normal vector reused) cannot be told from independent ones.
    D = DRModels.Distributions
    y_gam = [rand(rng, D.Gamma(4, exp(0.6 * η[i]) / 4)) for i in 1:n]
    y_beta = [rand(rng, D.Beta(p[i] * 5, (1 - p[i]) * 5)) for i in 1:n]
    y_nb = Int[rand(rng, D.Poisson(rand(rng, D.Gamma(3, exp(0.6 * η[i]) / 3)))) for i in 1:n]
    merge(d0, (; y_gam, y_beta, y_nb))
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

# --- between-group structure (see the header) -------------------------------------------
const _GS_NODES = QuadGK.gauss(64)          # Gauss-Legendre on [-1, 1]
function _gs_axis(s)
    s <= 0 && return [0.0], [1.0]
    x, w = _GS_NODES
    z = 8s .* x
    return z, w .* 8s .* exp.(-z .^ 2 ./ (2s^2)) ./ (s * sqrt(2π))
end

# Row covariance matrix C and row means of y for random-effect SDs (s1, s2) on (g, g2).
function _gs_rowcov(cm, η, σ, aux, s1, s2, g1, g2)
    z1, w1 = _gs_axis(s1); z2, w2 = _gs_axis(s2)
    n = length(η)
    M = Vector{Matrix{Float64}}(undef, n); Ev = zeros(n)
    for i in 1:n
        Mi = Matrix{Float64}(undef, length(z1), length(z2)); Vi = similar(Mi)
        for b in eachindex(z2), a in eachindex(z1)
            Mi[a, b], Vi[a, b] = cm(η[i] + z1[a] + z2[b], σ[i], aux[i])
        end
        M[i] = Mi; Ev[i] = dot(w1, Vi * w2)
    end
    Ei = [dot(w1, M[i] * w2) for i in 1:n]
    Ma = [M[i] * w2 for i in 1:n]            # E over the g2 effect, as a function of the g effect
    Mb = [M[i]' * w1 for i in 1:n]           # E over the g effect, as a function of the g2 effect
    C = zeros(n, n)
    for i in 1:n
        C[i, i] = Ev[i] + dot(w1, (M[i] .^ 2) * w2) - Ei[i]^2
        for j in i+1:n
            sg, sh = g1[i] == g1[j], g2[i] == g2[j]
            (sg || sh) || continue
            e = sg && sh ? dot(w1, (M[i] .* M[j]) * w2) :
                sg ? dot(w1, Ma[i] .* Ma[j]) : dot(w2, Mb[i] .* Mb[j])
            C[i, j] = C[j, i] = e - Ei[i] * Ei[j]
        end
    end
    return C, Ei
end

function _gs_stats(C, P1, P2)
    return (T1 = mean(diag(P1' * C * P1)), T2 = mean(diag(P2' * C * P2)), T3 = sum(P1' * C * P2))
end

function _check_groups(label, S, cm, η, σ, aux, sds, data; cf_scale = 0.7)
    n, N = size(S)
    lab1 = collect(getproperty(data, :g)); lab2 = collect(getproperty(data, :g2))
    g1 = indexin(lab1, unique(lab1)); g2 = indexin(lab2, unique(lab2))
    P1 = zeros(n, maximum(g1)); P2 = zeros(n, maximum(g2))
    for i in 1:n; P1[i, g1[i]] = 1; P2[i, g2[i]] = 1; end
    P1 ./= sum(P1; dims = 1); P2 ./= sum(P2; dims = 1)
    s1, s2 = Float64(sds[:g]), Float64(sds[:g2])
    C, Ei = _gs_rowcov(cm, η, σ, aux, s1, s2, g1, g2)
    T = _gs_stats(C, P1, P2)
    # draws: per-draw values whose mean is each statistic, and their standard error
    Y1 = P1' * S .- P1' * Ei; Y2 = P2' * S .- P2' * Ei
    zd = (T1 = vec(mean(abs2, Y1; dims = 1)), T2 = vec(mean(abs2, Y2; dims = 1)),
          T3 = vec(sum(Y1; dims = 1) .* sum(Y2; dims = 1)))
    est = map(mean, zd); bnd = map(z -> 5 * std(z) / sqrt(N), zd)
    # counterfactuals the checks must be able to tell from the truth (power): rows drawn
    # independently (diagonal C), and both SDs off by 30 percent
    Ti = _gs_stats(Diagonal(diag(C)), P1, P2)
    Cs, _ = _gs_rowcov(cm, η, σ, aux, cf_scale * s1, cf_scale * s2, g1, g2)
    Ts = _gs_stats(Cs, P1, P2)
    sep(Tc) = maximum(abs(Tc[k] - T[k]) / bnd[k] for k in (:T1, :T2, :T3))
    @info "$label group structure" T est bnd sep_rows_independent = sep(Ti) sep_sd_off = sep(Ts)
    for k in (:T1, :T2, :T3)
        @test abs(est[k] - T[k]) <= bnd[k]
    end
    @test sep(Ti) >= 2          # a per-row draw would sit at least 2 bounds (10 SE) away
    @test sep(Ts) >= 2          # SDs 30 percent too small: same
    return nothing
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
    _check_groups(label, S, cm, η, σ, aux, sds, data)
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
    _check_groups(label, S, (e, sg, nn) -> cm_trials(e, sg, nn), η, σ, trials, sds, data)
    return S
end
_cm_binomial(η, _σ, n) = (p = _lg(η); (n * p, n * p * (1 - p)))
_cm_betabin(η, σ, n) = (p = _lg(η); φ = 1 / σ^2; (n * p, n * p * (1 - p) * (φ + n) / (φ + 1)))

# A fitted object with `:trials` removed from `fit.scales` (what a fit saved by an earlier
# version carries), and a table whose columns raise an interrupt when read.
_strip_trials(fit) = DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov, fit.loglik,
    fit.nobs, fit.converged, fit.means, fit.obs,
    Dict{Symbol,Vector{Float64}}(k => v for (k, v) in fit.scales if k !== :trials),
    fit.formula, fit.nll, fit.nllgrad, fit.ranef, fit.estim_method, fit.reml_loglik,
    fit.ml_loglik, fit.marginal, fit.phylo_penalty, fit.penalty, fit.iterations,
    fit.phylo_scale, fit.uncertainty, fit.geometry)
struct _InterruptingTable end
Base.getproperty(::_InterruptingTable, ::Symbol) = throw(InterruptException())

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
        _check_moments("Gamma", drm(bf(f("y_gam"), s1), Gamma(); data = _SYN), _cm_gamma_prec; data = _SYN)
    end
    @testset "Beta" begin
        _check_moments("Beta", drm(bf(f("y_beta"), s1), Beta(); data = _SYN), _cm_beta; data = _SYN)
    end
    @testset "Poisson" begin
        _check_moments("Poisson", drm(bf(f("y_pois")), Poisson(); data = _SMM), _cm_poisson)
    end
    @testset "NegBinomial2" begin
        _check_moments("NB2", drm(bf(f("y_nb"), s1), NegBinomial2(); data = _SYN), _cm_nb2; data = _SYN)
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
    @testset "a fit that stored no trials: rebuilt from `data`, else refused" begin
        # A fit saved by an earlier version (or from a route that leaves `fit.scales` without
        # `:trials`, such as the three-intercept route on the combined tree): the simulator
        # rebuilds the trials from the response columns of `data`. The helper is exercised
        # here by removing `:trials` from a fitted object.
        fit = _strip_trials(drm(bf(f("cbind(succ, failc)")), Binomial(); data = _SYN))
        @test !haskey(fit.scales, :trials)
        tr = _SYN.succ .+ _SYN.failc
        _check_moments_trials("Binomial cbind, trials rebuilt", fit, tr, _cm_binomial; data = _SYN)
        fit01 = _strip_trials(drm(bf(f("y_bin")), Binomial(); data = _SYN))
        _check_moments_trials("Binomial 0/1, trials rebuilt", fit01, ones(fit01.nobs), _cm_binomial; data = _SYN)
        # response columns missing from `data`: trials cannot be known, so a clear refusal
        nores = (; g = _SYN.g, g2 = _SYN.g2, x1 = _SYN.x1)
        err = try; simulate(fit; data = nores); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("pass the fitting `data`", sprint(showerror, err))
        # the rebuild accepts what the fitter accepts and nothing else: failures that are
        # not whole numbers, or negative, are refused (the fitter refuses them too)
        for bad in (_SYN.failc .+ 0.5, _SYN.failc .- 100)
            e2 = try; simulate(fit; data = merge(_SYN, (; failc = bad))); nothing; catch e; e; end
            @test e2 isa ArgumentError
        end
        # an interrupt is never turned into a refusal
        @test_throws InterruptException DRModels._fit_with_simulate_trials(fit, _InterruptingTable())
    end
    @testset "rows with zero trials: the fit accepts them, so does the rebuild" begin
        z = 1:6
        zs = copy(_SYN.succ); zf = copy(_SYN.failc); zs[z] .= 0; zf[z] .= 0
        dz = merge(_SYN, (; succ = zs, failc = zf))
        fz = drm(bf(f("cbind(succ, failc)")), Binomial(); data = dz)
        @test fz.scales[:trials][z] == zeros(6)
        trz = zs .+ zf
        Sf = simulate(fz; nsim = 30, data = dz, rng = MersenneTwister(4))          # trials stored
        @test all(==(0), Sf[z, :])
        fzs = _strip_trials(fz)
        Sr = simulate(fzs; nsim = 30, data = dz, rng = MersenneTwister(4))         # trials rebuilt
        @test Sr == Sf                                   # same trials, same stream, same draws
        @test all(i -> all(0 .<= view(Sr, i, :) .<= trz[i]), axes(Sr, 1))
    end
    @testset "Binomial population-level draw (re = :zero) works too" begin
        # the refusal text offers `re = :zero`; it used to throw "does not carry `trials`"
        fit = drm(bf(f("cbind(succ, failc)")), Binomial(); data = _SYN)
        Z = simulate(fit; nsim = 5, re = :zero, rng = MersenneTwister(2))
        @test size(Z) == (fit.nobs, 5)
        @test all(isfinite, Z)
    end
end
