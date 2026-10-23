# #719: per-group adaptive Gauss–Hermite quadrature (AGHQ) for 1-D random
# intercepts `(1 | g)` on non-Gaussian families.
#
# Before #719 every `_fit_*_ranef` route integrated b_g on a fixed 32-node
# PRIOR-scale grid (b = √2 σ_b z), independent of where the group posterior mode
# actually sits. On an informative-group DGP the grid misses the posterior: for
# Poisson, `_RANEF1D_AGHQ_K` (K=5) is within 0.01 nat of an independent AGHQ-40+
# reference while the old 32-node PRIOR-scale grid is off by more than a nat (the
# `_old_prior_ghq_1d` reconstruction below reproduces that pre-#719 algorithm to
# demonstrate the failure — the production code no longer contains it).
#
# The reference here is an INDEPENDENT 1-D AGHQ implementation (ForwardDiff Newton
# to the group mode, eigendecomposition Gauss–Hermite nodes/weights), not the
# package's `_aghq_group_logint` helper — same discipline as test_adaptive_ghq.jl's
# `_ref_group834` for the 2-D (1 + x | g) routes.
using DRModels
using Test, Random, LinearAlgebra, ForwardDiff, DelimitedFiles, Statistics
import Distributions
const _Dd719 = Distributions

# --- independent 1-D reference -------------------------------------------------
function _gh719(K)
    K == 1 && return [0.0], [sqrt(π)]
    β = [sqrt(k / 2) for k in 1:(K-1)]
    E = eigen(SymTridiagonal(zeros(K), β))
    return E.values, sqrt(π) .* E.vectors[1, :] .^ 2
end

# log ∫ exp(h(b)) db for one group: mode by damped Newton, then K-node 1-D AGHQ.
function _ref_group719(h; K = 61)
    b = 0.0
    for _ in 1:200
        gr = ForwardDiff.derivative(h, b)
        Hm = -ForwardDiff.derivative(x -> ForwardDiff.derivative(h, x), b)
        δ = gr / Hm; t = 1.0
        while h(b + t * δ) < h(b) - 1e-12 && t > 1e-10
            t /= 2
        end
        b += t * δ
        abs(t * δ) < 1e-12 && break
    end
    Hm = -ForwardDiff.derivative(x -> ForwardDiff.derivative(h, x), b)
    σ = 1 / sqrt(Hm)
    z, w = _gh719(K); lw = log.(w)
    terms = [lw[k] + h(b + sqrt(2) * σ * z[k]) + z[k]^2 for k in 1:K]
    mx = maximum(terms)
    return log(2.0) / 2 + log(σ) + mx + log(sum(exp.(terms .- mx)))
end

# Pre-#719 algorithm (32-node Gauss–Hermite on the PRIOR scale b = √2 σ_b z), kept
# here only to demonstrate the gap this PR closes — no longer in `src/`.
function _old_prior_ghq_1d(ll, idx, σb; K = 32)
    z, w = DRModels._gauss_hermite(K); lw = log.(w); rt2 = sqrt(2.0); lπ = log(π)
    terms = Vector{Float64}(undef, K)
    for k in 1:K
        δ = rt2 * σb * z[k]
        terms[k] = lw[k] + sum(ll(i, δ) for i in idx)
    end
    mx = maximum(terms)
    return -0.5 * lπ + mx + log(sum(exp.(terms .- mx)))
end

# Package's current 1-D AGHQ helper at a given K (q = 1): same call the family
# fitters make in `_fit_*_ranef` (`η0 = zeros(n)`, `Zre = ones(n,1)`, since `ll`
# already has the group's fixed part baked in via `δ`).
function _pkg_group1d_full(ll, idx, n, σb; K)
    rule = DRModels._AGHQRule(1, K)
    Zre = ones(n, 1)
    return DRModels._aghq_group_logint(ll, idx, zeros(n), Zre, reshape([σb], 1, 1), rule, zeros(1))[1]
end

# Informative-group DGP (task spec): G = 100 groups, 30 obs/group, RE SD = 0.8.
function _dgp719(seed; G = 100, m = 30, sdb = 0.8)
    Random.seed!(seed)
    n = G * m
    g = repeat(1:G, inner = m)
    x = randn(n)
    b = sdb .* randn(G)
    return n, g, x, b
end

# For one family: build `ll(i, δ)` (δ = the group-offset argument the quadrature
# integrates over; the mean already has ηbase[i] folded in), then compare
# (a) the independent AGHQ-61+ reference, (b) the package helper at K = 1, 3,
# `_RANEF1D_AGHQ_K`, and (c) the pre-#719 32-node prior-scale grid, all summed
# over groups.
function _family_errors(ll_by_group, n, g, G; Ks = (1, 3, DRModels._RANEF1D_AGHQ_K), sdb = 0.8)
    idx_by_g = [findall(==(j), g) for j in 1:G]
    ref = 0.0; oldg = 0.0
    pkg = Dict(K => 0.0 for K in Ks)
    for j in 1:G
        idx = idx_by_g[j]
        isempty(idx) && continue
        ll = ll_by_group(j)
        h(b) = sum(ll(i, b) for i in idx) - 0.5 * b^2 / sdb^2 - 0.5 * log(2π) - log(sdb)
        ref += _ref_group719(h)
        oldg += _old_prior_ghq_1d(ll, idx, sdb)
        for K in Ks
            pkg[K] += _pkg_group1d_full(ll, idx, n, sdb; K = K)
        end
    end
    return ref, oldg, pkg
end

@testset "adaptive GHQ 1-D helper (#719): K = $(DRModels._RANEF1D_AGHQ_K) within 0.01 nat, per family" begin
    n, g, x, b = _dgp719(20260927)
    G = maximum(g)

    @testset "Poisson" begin
        η0 = 0.3 .+ 0.4 .* x
        y = Float64.([rand(_Dd719.Poisson(exp(η0[i] + b[g[i]]))) for i in 1:n])
        lf = [_Dd719.logfactorial(Int(v)) for v in y]
        ll_by_group = j -> (i, δ) -> (e = η0[i] + δ; y[i] * e - exp(e) - lf[i])
        ref, oldg, pkg = _family_errors(ll_by_group, n, g, G)
        @test abs(pkg[DRModels._RANEF1D_AGHQ_K] - ref) < 0.01
        @test abs(oldg - ref) > 0.3                      # pre-#719: fails badly (base bug, #719)
        @test abs(pkg[1] - ref) > 0.01                    # Laplace (K=1) is not enough here
        @test abs(pkg[3] - ref) > 0.01                    # K=3 undershoots the 0.01 nat bar
    end

    @testset "NegBinomial2" begin
        r = 4.0
        η0 = 0.3 .+ 0.4 .* x
        μ = exp.(η0 .+ b[g])
        y = Float64.([rand(_Dd719.NegativeBinomial(r, r / (r + μ[i]))) for i in 1:n])
        yint = round.(Int, y)
        ll_by_group = j -> (i, δ) -> (μi = exp(η0[i] + δ); p = r / (r + μi); _Dd719.logpdf(_Dd719.NegativeBinomial(r, p), yint[i]))
        ref, oldg, pkg = _family_errors(ll_by_group, n, g, G)
        @test abs(pkg[DRModels._RANEF1D_AGHQ_K] - ref) < 0.01
        @test abs(oldg - ref) > 0.3
    end

    @testset "Gamma" begin
        α = 3.0
        η0 = 0.3 .+ 0.2 .* x
        μ = exp.(η0 .+ b[g])
        y = Float64.([rand(_Dd719.Gamma(α, μ[i] / α)) for i in 1:n])
        ll_by_group = j -> (i, δ) -> (μi = exp(η0[i] + δ); _Dd719.logpdf(_Dd719.Gamma(α, μi / α), y[i]))
        ref, oldg, pkg = _family_errors(ll_by_group, n, g, G)
        @test abs(pkg[DRModels._RANEF1D_AGHQ_K] - ref) < 0.01
        @test abs(oldg - ref) > 0.05
    end

    @testset "Beta" begin
        φ = 15.0
        η0 = 0.2 .+ 0.3 .* x
        μ = 1 ./ (1 .+ exp.(-(η0 .+ b[g])))
        y = clamp.(Float64.([rand(_Dd719.Beta(μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n]), 1e-6, 1 - 1e-6)
        ll_by_group = j -> (i, δ) -> (μi = 1 / (1 + exp(-(η0[i] + δ))); _Dd719.logpdf(_Dd719.Beta(μi * φ, (1 - μi) * φ), y[i]))
        ref, oldg, pkg = _family_errors(ll_by_group, n, g, G)
        @test abs(pkg[DRModels._RANEF1D_AGHQ_K] - ref) < 0.01
        @test abs(oldg - ref) > 0.02
    end

    @testset "BetaBinomial" begin
        φ = 15.0; ntr = 20
        η0 = 0.2 .+ 0.3 .* x
        μ = 1 ./ (1 .+ exp.(-(η0 .+ b[g])))
        s = round.(Int, Float64.([rand(_Dd719.BetaBinomial(ntr, μ[i] * φ, (1 - μ[i]) * φ)) for i in 1:n]))
        ll_by_group = j -> (i, δ) -> (μi = 1 / (1 + exp(-(η0[i] + δ))); _Dd719.logpdf(_Dd719.BetaBinomial(ntr, μi * φ, (1 - μi) * φ), s[i]))
        ref, oldg, pkg = _family_errors(ll_by_group, n, g, G)
        @test abs(pkg[DRModels._RANEF1D_AGHQ_K] - ref) < 0.01
        @test abs(oldg - ref) > 0.02
    end

    @testset "Student" begin
        σ = 1.0; ν = 5.0
        η0 = 0.5 .+ 0.3 .* x
        y = Float64.([η0[i] + b[g[i]] + σ * rand(_Dd719.TDist(ν)) for i in 1:n])
        ll_by_group = j -> (i, δ) -> (μi = η0[i] + δ; zt = (y[i] - μi) / σ; _Dd719.logpdf(_Dd719.TDist(ν), zt) - log(σ))
        ref, oldg, pkg = _family_errors(ll_by_group, n, g, G)
        @test abs(pkg[DRModels._RANEF1D_AGHQ_K] - ref) < 0.01
    end

    @testset "LogNormal (exact at K = 1)" begin
        σ = 0.7
        η0 = 0.5 .+ 0.3 .* x
        ly = Float64.([η0[i] + b[g[i]] + σ * randn() for i in 1:n])
        ll_by_group = j -> (i, δ) -> _Dd719.logpdf(_Dd719.Normal(η0[i] + δ, σ), ly[i])
        ref, oldg, pkg = _family_errors(ll_by_group, n, g, G; Ks = (1,))
        @test abs(pkg[1] - ref) < 1e-6                    # linear-Gaussian marginal: K=1 is exact
    end
end

@testset "Poisson (1|g), HSAUR3::epilepsy (#719 issue reproduction)" begin
    dir = joinpath(@__DIR__, "fixtures", "adaptive_ghq_1d")
    raw, header = readdlm(joinpath(dir, "epilepsy.csv"), ','; header = true)
    col = Dict(String(header[1, j]) => j for j in 1:size(header, 2))
    y = Float64.(raw[:, col["y"]])
    trt = Float64.(raw[:, col["trt"]])
    logbase = Float64.(raw[:, col["logbase"]])
    subject = Int.(raw[:, col["subject"]])
    data = (; y, trt, logbase, subject)
    n = length(y)
    Xμ = hcat(ones(n), trt, logbase)
    gidx, G = DRModels._group_index(subject)

    # drmTMB 0.7.1, same data and formula (`poisson(link = "log")`, TMB Laplace,
    # i.e. K = 1 in this file's terms):
    #   fixef (Intercept, trt, logbase) = (-1.38545, -0.3354113, 1.01119)
    #   sd(subject) = 0.52369, logLik = -671.6764
    fit1 = DRModels._fit_poisson_ranef(Poisson(), y, Xμ, gidx, G, ["(Intercept)", "trt", "logbase"], :subject, 1e-10; K = 1)
    @test fit1.converged
    β1 = coef(fit1, :mu)
    @test β1[1] ≈ -1.38545 atol = 1e-3
    @test β1[2] ≈ -0.3354113 atol = 1e-3
    @test β1[3] ≈ 1.01119 atol = 1e-3
    @test re_sd(fit1)[:subject] ≈ 0.52369 atol = 1e-3
    @test loglik(fit1) ≈ -671.6764 atol = 1e-3         # K = 1 IS drmTMB's Laplace: matches to ~1e-3 nat

    # The package default (`drm`, K = `_RANEF1D_AGHQ_K` = 5): a strictly better
    # approximation to the TRUE marginal than TMB's own Laplace. Before #719 the
    # 32-node prior-scale grid put this optimum 5.2 nat below the true maximum
    # (issue #719); the AGHQ-5 default now sits within 0.01 nat of AGHQ-40+ truth
    # (see the informative-group sweep above), so its logLik legitimately differs
    # from drmTMB's Laplace figure by the intrinsic Laplace-vs-AGHQ gap, not a bug.
    fit5 = drm(bf(@formula(y ~ trt + logbase + (1 | subject))), Poisson(); data = data)
    @test fit5.converged
    β5 = coef(fit5, :mu)
    @test β5 ≈ β1 atol = 0.02                            # same optimum, small AGHQ-5-vs-Laplace shift
    @test re_sd(fit5)[:subject] ≈ re_sd(fit1)[:subject] atol = 0.02
    @test abs(loglik(fit5) - loglik(fit1)) < 0.2          # informative groups (59 subjects): a real but small AGHQ correction
end
