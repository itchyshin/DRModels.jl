# Fixed-effect meta-analytic route `y ~ x + meta_V(v)` (no random effect):
#   y_i ~ N(x_iᵀβ, v_i + σ²),  v_i known, σ the heterogeneity SD.
# Independent oracles, none of which reuse the fitter's objective:
# a hand-written profile likelihood over log σ (weighted least squares for β at
# each σ, golden-section search), the plain Gaussian fit for the constant-v and
# v → 0 limits, and a direct sum of Normal logpdfs.
using DRModels
using Test, Random, LinearAlgebra, Statistics
using Distributions: Normal, logpdf

# Profile NLL over s = log σ: β̂(s) is the closed-form weighted least squares fit
# with weights 1 / (v_i + e^{2s}).
function _mv_profile(s, y, X, v)
    w = 1 ./ (v .+ exp(2s))
    β = (X' * (w .* X)) \ (X' * (w .* y))
    r = y .- X * β
    return 0.5 * sum(log.(2π .* (v .+ exp(2s))) .+ w .* r .^ 2), β
end

function _mv_golden(f, a, b; tol = 1e-10)
    φ = (sqrt(5) - 1) / 2
    c, d = b - φ * (b - a), a + φ * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol
        if fc < fd
            b, d, fd = d, c, fc
            c = b - φ * (b - a); fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + φ * (b - a); fd = f(d)
        end
    end
    return (a + b) / 2
end

const _MV_BF = bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1))
const _MV_PLAIN = bf(@formula(y ~ x), @formula(sigma ~ 1))

function _mv_data(; k = 80, seed = 20261020, τ = 0.4)
    rng = MersenneTwister(seed)
    x = randn(rng, k)
    v = (0.2 .+ rand(rng, k)) .^ 2
    y = 0.3 .+ 0.5 .* x .+ τ .* randn(rng, k) .+ sqrt.(v) .* randn(rng, k)
    return (; y, x, v)
end

@testset "meta_V fixed route: independent oracles" begin
    d = _mv_data()
    X = [ones(length(d.y)) d.x]
    fit = drm(_MV_BF, Gaussian(); data = d)
    β̂ = coef(fit, :mu)
    ŝ = coef(fit, :sigma)[1]

    @testset "(1) profile-likelihood maximiser and WLS at fitted sigma" begin
        s★ = _mv_golden(s -> _mv_profile(s, d.y, X, d.v)[1], -6.0, 2.0)
        nll★, β★ = _mv_profile(s★, d.y, X, d.v)
        @test ŝ ≈ s★ atol = 1e-4
        @test β̂ ≈ β★ atol = 1e-5
        @test loglik(fit) ≈ -nll★ atol = 1e-6
        # closed-form WLS at the *fitted* sigma
        w = 1 ./ (d.v .+ exp(2ŝ))
        @test β̂ ≈ (X' * (w .* X)) \ (X' * (w .* d.y)) atol = 1e-6
    end

    @testset "(1b) intercept-only: inverse-variance weighted mean" begin
        d0 = (; y = d.y, v = d.v)
        f0 = drm(bf(@formula(y ~ 1 + meta_V(v)), @formula(sigma ~ 1)), Gaussian(); data = d0)
        w = 1 ./ (d.v .+ exp(2 * coef(f0, :sigma)[1]))
        @test coef(f0, :mu)[1] ≈ sum(w .* d.y) / sum(w) atol = 1e-6
        s★ = _mv_golden(s -> _mv_profile(s, d.y, ones(length(d.y), 1), d.v)[1], -6.0, 2.0)
        @test coef(f0, :sigma)[1] ≈ s★ atol = 1e-4
    end

    @testset "(4) loglik is the sum of independent Normal logpdfs" begin
        μ = X * β̂
        sd = sqrt.(d.v .+ exp(2ŝ))
        @test loglik(fit) ≈ sum(logpdf.(Normal.(μ, sd), d.y)) atol = 1e-8
    end

    @testset "(5) vcov / stderror finite and symmetric" begin
        V = vcov(fit)
        @test size(V) == (3, 3)
        @test all(isfinite, V)
        @test V ≈ V' atol = 1e-12
        @test all(>(0), diag(V))
        se = stderror(fit)
        @test all(isfinite, se) && all(>(0), se)
        @test se ≈ sqrt.(diag(V)) atol = 1e-10
    end

    @testset "(7) fitted and predict equal Xb" begin
        @test fitted(fit) ≈ X * β̂ atol = 1e-10
        nd = (; x = [-1.0, 0.0, 2.5])
        @test predict(fit, nd) ≈ β̂[1] .+ β̂[2] .* nd.x atol = 1e-10
        @test predict(fit, d) ≈ X * β̂ atol = 1e-10
    end
end

@testset "meta_V fixed route: limits against the plain Gaussian fit" begin
    d = _mv_data(seed = 7)
    k = length(d.y)

    @testset "(2) constant v = c: variance is c + sigma^2" begin
        c = 0.25
        dc = (; d.y, d.x, v = fill(c, k))
        fm = drm(_MV_BF, Gaussian(); data = dc)
        fp = drm(_MV_PLAIN, Gaussian(); data = dc)
        σ²_plain = exp(2 * coef(fp, :sigma)[1])
        # the ML total variance is identified; the split c + σ² needs σ̂² > 0
        @test σ²_plain > c
        @test exp(2 * coef(fm, :sigma)[1]) + c ≈ σ²_plain rtol = 1e-4
        @test coef(fm, :mu) ≈ coef(fp, :mu) atol = 1e-5
        @test loglik(fm) ≈ loglik(fp) atol = 1e-6
    end

    @testset "(3) v -> 0 recovers the plain Gaussian fit" begin
        d0 = (; d.y, d.x, v = fill(1e-10, k))
        fm = drm(_MV_BF, Gaussian(); data = d0)
        fp = drm(_MV_PLAIN, Gaussian(); data = d0)
        @test coef(fm, :mu) ≈ coef(fp, :mu) atol = 1e-5
        @test coef(fm, :sigma)[1] ≈ coef(fp, :sigma)[1] atol = 1e-4
        @test loglik(fm) ≈ loglik(fp) atol = 1e-6
    end
end

# (6) REML: the fixed meta_V route refuses `method = :REML` on this build, so
# there is no REML-vs-ML sigma ordering to test; deliberately not pinned here
# (the refusal wording is owned by the input-validation lane).
