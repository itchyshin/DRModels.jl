# test_betabinomial_crossed_laplace.jl — public-API recovery + a low-level
# exact FD-vs-analytic ≤ 1e-6 gate for BetaBinomial()'s crossed random-
# intercept `(1 | g) + (1 | h)` sparse-Laplace route (#166), constant-σ
# (overdispersion) only. Mirrors test_crossed_laplace_generic.jl's shape.

using DRModels
using Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma, digamma

_bbcr_logistic(x) = 1 / (1 + exp(-x))

@testset "BetaBinomial crossed random intercepts - sparse Laplace route" begin
    # G/H/n sized like test_crossed_laplace_generic.jl (G=28, H=24, n=2400) —
    # a handful of groups makes ML variance-component recovery both
    # small-sample-biased and sensitive to platform-level floating-point
    # noise (BLAS/libm) propagating through the Laplace Newton iterations;
    # this scale keeps both σg/σh comfortably inside tolerance across
    # macOS/Linux and Julia 1.10/1.12.
    rng = MersenneTwister(20260804)
    G = 28
    H = 24
    n = 2400
    x = randn(rng, n)
    gids = [rand(rng, 1:G) for _ in 1:n]
    hids = [rand(rng, 1:H) for _ in 1:n]
    g = [Symbol("g", j) for j in gids]
    h = [Symbol("h", j) for j in hids]
    β = [0.10, 0.40]
    σg = 0.35
    σh = 0.25
    bg = σg .* randn(rng, G)
    bh = σh .* randn(rng, H)
    bg .-= sum(bg) / G
    bh .-= sum(bh) / H
    η = [β[1] + β[2] * x[i] + bg[gids[i]] + bh[hids[i]] for i in 1:n]
    μ = _bbcr_logistic.(η)
    precision = 18.0
    ntr = fill(8, n)
    successes = Float64.([rand(rng, Distributions.BetaBinomial(ntr[i], μ[i] * precision, (1 - μ[i]) * precision)) for i in 1:n])
    failures = Float64.(ntr) .- successes

    fit = drm(bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
              BetaBinomial(); data = (; successes, failures, x, g, h))

    @test fit.converged
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.12
    @test haskey(re_sd(fit), :g)
    @test haskey(re_sd(fit), :h)
    @test abs(re_sd(fit)[:g] - σg) < 0.15
    @test abs(re_sd(fit)[:h] - σh) < 0.15
    @test isfinite(loglik(fit))
    @test all(0 .< fitted(fit) .< 1)

    # Nonconstant-sigma is out of scope for #166 — must error.
    @test_throws ErrorException drm(
        bf(@formula(cbind(successes, failures) ~ x + (1 | g) + (1 | h)), @formula(sigma ~ x)),
        BetaBinomial(); data = (; successes, failures, x, g, h)
    )
end

@testset "BetaBinomial crossed sparse-Laplace nuisance exact gradient" begin
    rng = MersenneTwister(20260805)
    G = 6
    H = 5
    n = 220
    x = randn(rng, n)
    gidx = [rand(rng, 1:G) for _ in 1:n]
    hidx = [rand(rng, 1:H) for _ in 1:n]
    X = hcat(ones(n), x)
    β = [0.15, 0.35]
    bg = 0.35 .* randn(rng, G)
    bh = 0.25 .* randn(rng, H)
    η = [β[1] + β[2] * x[i] + bg[gidx[i]] + bh[hidx[i]] for i in 1:n]
    μ = _bbcr_logistic.(η)

    ntr = fill(8, n)
    sint = [rand(rng, Distributions.BetaBinomial(ntr[i], μ[i] * 20.0, (1 - μ[i]) * 20.0)) for i in 1:n]
    logchoose = [DRModels._logfactorial(ntr[i]) - DRModels._logfactorial(sint[i]) -
                 DRModels._logfactorial(ntr[i] - sint[i]) for i in 1:n]
    function aux_from(logsigma)
        φ = exp(clamp(-2 * logsigma, -8.0, 8.0))
        lgamma_nphi = [loggamma(ntr[i] + φ) for i in 1:n]
        return (s = sint, ntr = ntr, logchoose = logchoose, precision = φ,
                lgamma_nphi = lgamma_nphi, lgammaφ = loggamma(φ), digammaφ = digamma(φ))
    end

    θ = [0.12, 0.32, -0.5 * log(22.0), log(0.32), log(0.24)]
    val, grad, _, ok = DRModels._crossed_mean_laplace_nuisance_fg(
        Val(:betabinomial_fixed), aux_from, n, X, gidx, G, hidx, H, θ; grad = true
    )
    @test ok
    @test isfinite(val)

    function central_gradient(f, θ0; h = 1e-5)
        g = similar(θ0)
        for k in eachindex(θ0)
            step = h * max(abs(θ0[k]), 1.0)
            θp = copy(θ0); θm = copy(θ0)
            θp[k] += step; θm[k] -= step
            g[k] = (f(θp) - f(θm)) / (2step)
        end
        return g
    end
    fd = central_gradient(θp -> DRModels._crossed_mean_laplace_nuisance_fg(
        Val(:betabinomial_fixed), aux_from, n, X, gidx, G, hidx, H, θp; grad = false
    )[1], θ)
    max_abs_diff = maximum(abs.(grad .- fd))
    @info "BetaBinomial crossed gradient gate" max_abs_diff grad fd
    @test max_abs_diff ≤ 1e-6
end
