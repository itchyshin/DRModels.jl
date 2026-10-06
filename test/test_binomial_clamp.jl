# Binomial objective must not be clamped: a clamp of the linear predictor inside the
# objective makes it exactly flat (zero gradient) wherever |η| exceeds the clamp, so
# L-BFGS can overshoot onto the plateau and report convergence with a garbage fit.
# Reference: Newton/IRLS with step-halving on the exact (log1pexp) log-likelihood.
using DRModels
using Test, LinearAlgebra
import Distributions

_l1pe(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
_ref_ll(β, X, k, n) = sum(k .* (X * β) .- n .* _l1pe.(X * β))
function _ref_mle(X, k, n)
    β = zeros(size(X, 2))
    for _ in 1:200
        η = X * β; μ = 1 ./ (1 .+ exp.(-η))
        g = X' * (k .- n .* μ); H = X' * (X .* (n .* μ .* (1 .- μ)))
        st = H \ g; t = 1.0
        while _ref_ll(β .+ t .* st, X, k, n) < _ref_ll(β, X, k, n) - 1e-14 && t > 1e-8
            t /= 2
        end
        β = β .+ t .* st
        norm(g) < 1e-12 && break
    end
    return β
end

# Non-separated Bernoulli data (the single overlap is x = 4.83 with y = 0, between
# y = 1 at x = 1.5 and 5.51). True MLE slope ≈ 0.70; main (clamped) returned slope
# ≈ 50.8, loglik = -15 (the clamp plateau) with converged = true.
const _X262 = [10.72, 8.04, -4.3, -0.82, 1.63, 1.53, 1.5, -2.21, -7.99, -2.75, -9.06, -3.64,
               -2.45, -6.5, -2.2, 5.51, -3.91, 14.69, 1.91, 4.83, 8.83, 4.65]
const _Y262 = Float64[1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1]
const _X183 = [-5.59, -1.51, 3.85, 4.31, -2.09, -8.0, -1.09, -11.15, 23.34, -9.65, 5.07, 9.99, -2.5,
               10.02, -16.76, -2.2, -16.09, 13.34, -11.24, -12.52, 19.75, -3.91, 11.8, 5.93, 0.66,
               -7.26, -0.95, -6.14, -5.73]
const _Y183 = Float64[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]

@testset "Binomial FE: no clamp plateau (overlapping outliers, wide x)" begin
    for (x, y) in ((_X262, _Y262), (_X183, _Y183))
        X = [ones(length(x)) x]; n1 = ones(length(x))
        β̂ = _ref_mle(X, y, n1)
        fit = drm(bf(@formula(y ~ x)), Binomial(); data = (; y, x))
        @test coef(fit, :mu) ≈ β̂ atol = 1e-6
        @test loglik(fit) ≈ _ref_ll(β̂, X, y, n1) atol = 1e-6
        @test fit.converged
    end
end

@testset "Binomial cbind(s, f) FE: grouped trials, wide x, no plateau" begin
    x = _X183; n = fill(8.0, length(x))
    k = Float64.(round.(8 .* ifelse.(_Y183 .== 1, 0.85, 0.1)))   # mostly-ordered counts
    k[3] = 0.0; k[15] = 8.0                                       # two overlapping outliers
    d = (; s = k, fl = n .- k, x)
    X = [ones(length(x)) x]
    β̂ = _ref_mle(X, k, n)
    fit = drm(bf(@formula(cbind(s, fl) ~ x)), Binomial(); data = d)
    @test coef(fit, :mu) ≈ β̂ atol = 1e-6
    @test loglik(fit) ≈ _ref_ll(β̂, X, k, n) + sum(DRModels._logchoose.(n, k)) atol = 1e-6
end

@testset "Binomial log-likelihood kernel matches Distributions off the plateau" begin
    for nt in (1, 7, 40), k in (0, 1, nt ÷ 2, nt), η in (-14.0, -3.0, -0.2, 0.0, 0.9, 6.0, 14.0)
        k > nt && continue
        lc = DRModels._logchoose(nt, k)
        @test DRModels._binomial_logit_ll(nt, k, η, lc) ≈
              Distributions.logpdf(Distributions.Binomial(nt, 1 / (1 + exp(-η))), k) atol = 1e-8   # the reference itself loses ~1e-10 to logistic saturation
    end
    # finite and exact far beyond the old ±15 clamp
    @test isfinite(DRModels._binomial_logit_ll(1, 0, 800.0, 0.0))
    @test DRModels._binomial_logit_ll(1, 0, 800.0, 0.0) ≈ -800.0
    @test DRModels._binomial_logit_ll(1, 1, -800.0, 0.0) ≈ -800.0
end

# Random-intercept route (adaptive Gauss-Hermite): Bernoulli data, 5 groups, wide x.
# main (clamped): coefficient ≈ 868 stuck at the plateau, loglik = -15, converged = true.
@testset "Binomial (1 | g): fit not stuck on the plateau" begin
    x = [6.55, 4.94, 5.07, -5.41, -2.0, 8.55, 7.54, 0.95, 1.41, -7.5, -6.69, -9.23, -3.99, 10.65, -12.55,
         -1.76, -7.53, -1.16, 3.39, 3.38, 4.65, 1.72, -0.14, -2.58, -2.64]
    y = Float64[1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    g = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5]
    fit = drm(bf(@formula(y ~ x + (1 | g))), Binomial(); data = (; y, x, g))
    X = [ones(length(x)) x]
    ll_fe = _ref_ll(_ref_mle(X, y, ones(length(x))), X, y, ones(length(x)))
    # a random intercept can only add marginal likelihood over the FE model (σ_b → 0)
    @test loglik(fit) > ll_fe - 0.05
    @test loglik(fit) > -2.84                      # measured -2.8325 (main: -15.0)
    @test maximum(abs, coef(fit, :mu)) < 10        # main: 868
end

# BetaBinomial shared the ±15 logit clamp on the (1 | g) route and a ±30 clamp on the FE route.
@testset "BetaBinomial FE: near-separated, no plateau" begin
    x = [2.55, -6.63, 3.73, 4.3, 14.5, -4.05, 2.62, -1.17, 2.96, -2.64, -5.7, -2.44, 1.32, -3.78, 8.51,
         -7.61, 5.44, 2.35, -4.53, 3.38]
    k = [10, 0, 10, 10, 10, 0, 10, 1, 10, 0, 0, 0, 10, 0, 10, 0, 10, 10, 0, 10]
    fit = drm(bf(@formula(cbind(s, fl) ~ x), @formula(sigma ~ 1)), BetaBinomial();
              data = (; s = Float64.(k), fl = Float64.(10 .- k), x))
    @test loglik(fit) > -1.0                       # measured -0.946 (main: -59.9, converged = true)
    # Near separation leaves a gradient around 1e-3, so the gradient criterion
    # does not fire at g_tol = 1e-8. The likelihood guard above is the plateau check.
end

@testset "BetaBinomial log-likelihood kernel matches Distributions off the plateau" begin
    for nt in (1, 10), k in (0, 1, nt), η in (-6.0, -0.3, 0.0, 1.2, 6.0), φ in (0.5, 4.0, 60.0)
        μ = 1 / (1 + exp(-η))
        @test DRModels._betabinomial_logit_ll(nt, k, η, φ) ≈
              Distributions.logpdf(Distributions.BetaBinomial(nt, μ * φ, (1 - μ) * φ), k) atol = 1e-8
    end
    @test isfinite(DRModels._betabinomial_logit_ll(10, 3, 5000.0, 4.0))     # guard, no loggamma(0)
end

@testset "BetaBinomial (1 | g): no plateau" begin
    x = [18.2, 16.2, -37.69, -20.73, 35.89, -21.52, -19.92, -4.2, -3.58, -37.51, -12.53, 25.17, 5.17, -28.49,
         36.92, -9.28, 14.69, 44.02, -9.87, -3.51, -8.43, -21.86, 7.29, 16.42, -9.4, -2.88, -15.79, 7.73,
         -8.21, -38.25, -11.65, -28.65, -19.13, 13.68]
    k = [10, 10, 0, 0, 10, 0, 0, 2, 2, 0, 0, 10, 9, 0, 10, 0, 10, 10, 0, 3, 0, 0, 10, 10, 0, 2, 0, 10, 0, 0, 0, 0, 0, 10]
    g = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
    fit = drm(bf(@formula(cbind(s, fl) ~ x + (1 | g)), @formula(sigma ~ 1)), BetaBinomial();
              data = (; s = Float64.(k), fl = Float64.(10 .- k), x, g))
    @test loglik(fit) > -7.7                       # measured -7.691 (main: -73.4, converged = true)
end

# Lifting the clamp must not break fits whose MLE does not exist: completely separated
# data still terminates (iteration cap / gradient tolerance) and returns a fit.
@testset "Binomial: completely separated data still terminates with a fit" begin
    x = collect(-5.0:1.0:5.0); x = x[x .!= 0]; y = Float64.(x .> 0)
    fit = drm(bf(@formula(y ~ x)), Binomial(); data = (; y, x))
    @test isfinite(loglik(fit))
    @test loglik(fit) > -1e-3                      # likelihood sup is 0
    @test coef(fit, :mu)[2] > 5                    # slope diverges toward +∞
end

# HagerZhang asserted (`B > A`) on this draw once the clamp was gone and the fit threw;
# main returned a non-converged fit. It must return a fit (converged or honestly not).
@testset "BetaBinomial (1 | g): line-search assertion falls back, never throws" begin
    x = [-4.87, 0.64, -10.52, -4.51, 5.23, 6.68, 5.81, 1.56, -5.13, 9.52, -9.16, 18.52, -3.42, -10.57, 7.85,
         4.47, 12.18, -11.08, 1.53, -0.21, -5.96]
    k = [1, 6, 1, 0, 9, 10, 10, 6, 1, 10, 0, 10, 1, 1, 9, 10, 10, 0, 8, 4, 1]
    g = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5]
    fit = drm(bf(@formula(cbind(s, fl) ~ x + (1 | g)), @formula(sigma ~ 1)), BetaBinomial();
              data = (; s = Float64.(k), fl = Float64.(10 .- k), x, g))
    @test fit isa DRModels.DrmFit
    @test isfinite(loglik(fit))
    @test loglik(fit) > -25                        # main: -21.66 (non-converged)
end
