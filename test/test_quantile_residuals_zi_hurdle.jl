# Randomized quantile residuals of zero-inflated and hurdle count fits must use
# the FULL mixture CDF (#922), not the count component alone.
#
#   zero-inflated: F(y) = π + (1 − π)·F_c(y),                       y ≥ 0
#   hurdle:        F(0) = p0,  F(y) = p0 + (1 − p0)·(F_c(y) − F_c(0)) / (1 − F_c(0)),  y ≥ 1
#
# with F(−1) = 0, π = scales[:zi], p0 = scales[:hu], F_c the Poisson / NB2 CDF at
# the fitted count mean `means[:mu]`. Before the fix the PIT was built from F_c
# alone, so under a CORRECT model the residuals were far from N(0, 1): on the four
# datasets below √n·D was 13.0 (ZI Poisson), 12.0 (ZI NB2), 9.3 (hurdle Poisson)
# and 6.2 (hurdle NB2), with means −0.64 … −0.26.
using DRModels
using Test, Statistics, StableRNGs
import Distributions

# √n · sup_x |F_n(x) − Φ(x)|; under H0 this follows the Kolmogorov distribution,
# and 1.7 is roughly α ≈ 0.006.
function _qzh_ks_sqrtn(r)
    n = length(r); s = sort(r); D = 0.0
    for i in 1:n
        F = Distributions.cdf(Distributions.Normal(), s[i])
        D = max(D, abs(i / n - F), abs(F - (i - 1) / n))
    end
    return sqrt(n) * D
end

# Count component at mean λ: Poisson, or NB2 with size r (r === nothing → Poisson).
_qzh_count(λ, r) = r === nothing ? Distributions.Poisson(λ) :
                   Distributions.NegativeBinomial(r, r / (r + λ))

# Brute-force mixture CDF: the sum of the mixture pmf over 0:y (Distributions'
# pmf only, independent of the package's CDF code).
function _qzh_mixture_cdf(kind, y, λ, r, p)
    y < 0 && return 0.0
    c = _qzh_count(λ, r)
    s = 0.0
    for k in 0:y
        pk = kind === :zi ? (k == 0 ? p : 0.0) + (1 - p) * Distributions.pdf(c, k) :
             k == 0 ? p : (1 - p) * Distributions.pdf(c, k) / (1 - Distributions.pdf(c, 0))
        s += pk
    end
    return s
end

# Simulate y from the true zero-inflated (`:zi`) or hurdle (`:hu`) model:
# log λ = 0.6 + 0.4x, logit p = −0.4 + 0.8w, NB2 size 2.
function _qzh_simulate(kind, fam, n, seed)
    rng = StableRNG(seed)
    x = randn(rng, n); w = randn(rng, n)
    λ = exp.(0.6 .+ 0.4 .* x)
    p = 1 ./ (1 .+ exp.(-(-0.4 .+ 0.8 .* w)))
    y = zeros(n)
    for i in 1:n
        c = _qzh_count(λ[i], fam === :pois ? nothing : 2.0)
        if rand(rng) < p[i]
            y[i] = 0
        elseif kind === :zi
            y[i] = rand(rng, c)
        else
            k = 0
            while k == 0
                k = rand(rng, c)
            end
            y[i] = k
        end
    end
    return (; y, x, w)
end

function _qzh_fit(kind, fam, dat)
    zpart = kind === :zi ? @formula(zi ~ w) : @formula(hu ~ w)
    return fam === :pois ?
        drm(bf(@formula(y ~ x), zpart), DRModels.Poisson(); data = dat) :
        drm(bf(@formula(y ~ x), @formula(sigma ~ 1), zpart), NegBinomial2(); data = dat)
end

@testset "ZI / hurdle quantile residuals use the mixture CDF (#922)" begin
    n = 2000
    for (kind, fam, seed) in ((:zi, :pois, 9221), (:zi, :nb2, 9222),
                              (:hu, :pois, 9223), (:hu, :nb2, 9224))
        @testset "$kind $fam" begin
            dat = _qzh_simulate(kind, fam, n, seed)
            fit = _qzh_fit(kind, fam, dat)
            @test fit.converged

            # Under the true model the residuals are ≈ N(0, 1).
            r = residuals(fit; type = :quantile, rng = StableRNG(1))
            @test length(r) == n
            @test all(isfinite, r)
            @test _qzh_ks_sqrtn(r) < 1.7
            @test abs(mean(r)) < 0.1
            @test 0.9 < std(r) < 1.1

            # Exact: replaying the same uniforms (one draw per row, in row order)
            # through the brute-force mixture CDF reproduces every PIT value.
            λ̂ = fitted(fit); p̂ = fit.scales[kind]
            rsize = fam === :nb2 ? 1 ./ fit.scales[:sigma] .^ 2 : fill(nothing, n)
            urng = StableRNG(1)
            uref = map(1:n) do i
                yi = round(Int, dat.y[i])
                a = _qzh_mixture_cdf(kind, yi - 1, λ̂[i], rsize[i], p̂[i])
                b = _qzh_mixture_cdf(kind, yi, λ̂[i], rsize[i], p̂[i])
                clamp(a + (b - a) * rand(urng), eps(), 1 - eps())
            end
            @test maximum(abs.(Distributions.cdf.(Distributions.Normal(), r) .- uref)) < 1e-10

            # Exact: the PIT interval [F(y − 1), F(y)] equals the brute-force
            # mixture CDF on a grid of y, including y = 0 and the far upper tail.
            err = 0.0
            for i in (1, 17, 523, 1999), y in 0:40
                a, b = DRModels._zi_hurdle_pit_interval(fit, fit.family, i, y;
                                                        μ = λ̂, gsis = false, mix = nothing)
                err = max(err, abs(a - _qzh_mixture_cdf(kind, y - 1, λ̂[i], rsize[i], p̂[i])),
                               abs(b - _qzh_mixture_cdf(kind, y, λ̂[i], rsize[i], p̂[i])))
            end
            @test err < 1e-12
        end
    end

    # Hurdle NB2 at extreme dispersion (size r = 1/σ² ≈ 1e17–1e43, where
    # Distributions' NB2 CDF rounds to 1): the positive part reuses the log-space
    # zero-truncated NB2 CDF, so the PIT stays finite and matches the r → ∞
    # limit, the hurdle Poisson.
    @testset "hurdle NB2 at extreme dispersion" begin
        y = [0.0, 1.0, 2.0, 5.0, 10.0]; m = length(y)
        fake(fam, μ, scales) = DRModels.DrmFit(fam, Pair{Symbol,UnitRange{Int}}[:mu => 1:1],
            Pair{Symbol,Vector{String}}[:mu => ["(Intercept)"]], [0.0], fill(NaN, 1, 1), NaN, m, true,
            Dict(:mu => μ), Dict(:mu => y), scales)
        for logσ in (-20.0, -30.0, -50.0), μ0 in (0.5, 5.0, 50.0)
            μ = fill(μ0, m)
            fnb = fake(NegBinomial2(), μ, Dict(:sigma => fill(exp(logσ), m), :hu => fill(0.3, m)))
            fpo = fake(DRModels.Poisson(), μ, Dict(:hu => fill(0.3, m)))
            @test all(isfinite, DRModels._quantile_residuals(fnb, StableRNG(3)))
            for i in 1:m
                yi = round(Int, y[i])
                abnb = DRModels._zi_hurdle_pit_interval(fnb, fnb.family, i, yi; μ = μ, gsis = false, mix = nothing)
                abpo = DRModels._zi_hurdle_pit_interval(fpo, fpo.family, i, yi; μ = μ, gsis = false, mix = nothing)
                @test all(isfinite, abnb)
                @test abnb[1] ≈ abpo[1] atol = 1e-10
                @test abnb[2] ≈ abpo[2] atol = 1e-10
            end
        end

        # A count mean that underflowed to exactly 0: the zero-truncated part
        # puts all its mass at y = 1, so the hurdle PIT interval at y = 1 is
        # [p0, 1] for NB2 exactly as for Poisson (not the saturated [1, 1]).
        μ = zeros(m)
        for logσ in (-1.0, -30.0)
            fnb = fake(NegBinomial2(), μ, Dict(:sigma => fill(exp(logσ), m), :hu => fill(0.3, m)))
            fpo = fake(DRModels.Poisson(), μ, Dict(:hu => fill(0.3, m)))
            abnb = DRModels._zi_hurdle_pit_interval(fnb, fnb.family, 2, 1; μ = μ, gsis = false, mix = nothing)
            abpo = DRModels._zi_hurdle_pit_interval(fpo, fpo.family, 2, 1; μ = μ, gsis = false, mix = nothing)
            @test abnb[1] ≈ 0.3 atol = 1e-12
            @test abnb[2] ≈ 1.0 atol = 1e-12
            @test abnb[1] ≈ abpo[1] atol = 1e-12
            @test abnb[2] ≈ abpo[2] atol = 1e-12
            @test all(isfinite, DRModels._quantile_residuals(fnb, StableRNG(4)))
        end
    end
end
