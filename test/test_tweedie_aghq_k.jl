# test_tweedie_aghq_k.jl — pins the K sweep behind `_TWEEDIE_RANEF_AGHQ_K` (#881,
# post-review). Review of the AGHQ switch found the generic default
# `_RANEF1D_AGHQ_K = 5` still 0.19 nat off `_fit_tweedie_ranef`'s logLik at θ̂ on
# a zero-heavy, informative-group DGP, even though it is within 1e-6 nat on the
# single-group σ_b=6 demo in test_tweedie_aghq.jl. Sweeping K ∈ {5, 9, 15, 21}
# (following #877's precedent of a family-specific K constant,
# `_BINOMIAL_RANEF_AGHQ_K`) on the review's own zero-heavy probe and on the
# review's "test (b)" DGP: K=21 is the smallest of the four with worst-case
# error ≤ 1e-3 nat on BOTH. `_TWEEDIE_RANEF_AGHQ_K` (src/tweedie.jl) is set to
# 21 and is now the default `K` for `_fit_tweedie_ranef`/`_fit_tweedie_slope_ranef`.
#
# This test pins: (a) the constant's value; (b) the default-K fit on the
# zero-heavy DGP matches an independent QuadGK total at θ̂ within 1e-3 nat
# (K=5 misses this bar at ~0.19 nat, measured by hand against the review's own
# probe script).
module TestTweedieAGHQK

using DRModels
using Test, StableRNGs, QuadGK
import Distributions as D

@testset "_TWEEDIE_RANEF_AGHQ_K: value and zero-heavy accuracy" begin
    @test DRModels._TWEEDIE_RANEF_AGHQ_K == 21

    # Independent per-group reference: crude mode-by-grid then QuadGK, matching
    # the review's own probe881.jl exactly (so this reproduces its numbers).
    function exact_g(y, eta0, xs, phi, p, sb)
        h(b) = sum(DRModels._logpdf_tweedie(y[i], exp(clamp(eta0[i] + b * xs[i], -30.0, 30.0)), phi, p)
                   for i in eachindex(y)) + D.logpdf(D.Normal(0, sb), b)
        grid = range(-12sb, 12sb; length = 4001); bs = grid[argmax(h.(grid))]
        sh = h(bs)
        v, _ = quadgk(b -> exp(h(b) - sh), -12sb, bs, 12sb; rtol = 1e-12, order = 15)
        return log(v) + sh
    end
    function exact_total(fit, y, X, xs, g, G)
        β = coef(fit, :mu); φ = exp(2 * coef(fit, :sigma)[1])
        p = 1 + 1 / (1 + exp(-coef(fit, :nu)[1])); sb = exp(coef(fit, :resd)[1])
        e0 = X * β
        return sum(exact_g(y[g .== j], e0[g .== j], xs[g .== j], φ, p, sb) for j in 1:G)
    end

    # Zero-heavy, informative-group DGP from the review's probe881.jl:
    # G=60, m=5, sigma_b=3, beta0=-1.5, zero probability exp(-mu) per row.
    rng = StableRNG(5)
    G = 60; m = 5; n = G * m; sb_dgp = 3.0; beta0 = -1.5
    g = repeat(1:G, inner = m)
    x = randn(rng, n)
    b = sb_dgp .* randn(rng, G)
    mu = exp.(beta0 .+ 0.4 .* x .+ b[g])
    y = [rand(rng) < exp(-mu[i]) ? 0.0 : rand(rng, D.Gamma(2.0, mu[i] / 2)) for i in 1:n]
    gidx, G_ = DRModels._group_index(string.(g))
    X = hcat(ones(n), x)
    args = (Tweedie(), y, X, ones(n, 1), ones(n, 1), gidx, G_,
            ["(Intercept)", "x"], ["(Intercept)"], ["(Intercept)"], :id, 1e-8)

    fit_default = DRModels._fit_tweedie_ranef(args...)   # default K = _TWEEDIE_RANEF_AGHQ_K = 21
    ex = exact_total(fit_default, y, X, ones(n), g, G_)
    @test loglik(fit_default) ≈ ex atol = 1e-3   # measured 6.7e-4 nat at K=21

    fit_k5 = DRModels._fit_tweedie_ranef(args...; K = 5)
    ex5 = exact_total(fit_k5, y, X, ones(n), g, G_)
    @test abs(loglik(fit_k5) - ex5) > 1e-2   # K=5 misses the 1e-3 bar here (measured ~0.19 nat), non-vacuous
end

end # module
