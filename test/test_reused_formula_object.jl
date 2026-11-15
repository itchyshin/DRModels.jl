# One bf(...) object reused for a second fit equals a fresh one (P31).
#
# A formula object used to keep the training schema of its first fit. Reusing it
# then failed in two ways on the plain Gaussian route: a response that is
# Vector{Union{Missing,Float64}} with no missing value, followed by data with
# missing values, stopped with "predict: could not build the `mu` design matrix"
# (KeyError); and a second data set with different factor levels stopped with
# "newdata column `f` contains level(s) ... not seen when the model was fitted",
# in both directions and for `mu` and `sigma`. Each fit now works on its own
# copy of the formula with an empty cache, so a reused object must give the same
# fit as a fresh one. The Poisson, random-intercept and bivariate cases were
# already right and stay as guards against a regression.
#
# Every reused-formula fit is compared with a fresh-formula fit on the same data
# by `==` on coef, loglik and nobs. Seeded Xoshiro data, fixed in this file.
using DRModels
using Test
using Random
using Logging

_reuse_quiet(f) = Logging.with_logger(f, Logging.NullLogger())
_reuse_same(a, b) = coef(a) == coef(b) && loglik(a) == loglik(b) && nobs(a) == nobs(b)

@testset "a reused formula object equals a fresh one (stale schema)" begin
    rng = Xoshiro(31); n = 120
    x = randn(rng, n); x2 = randn(rng, n)
    g = repeat(string.('a':'l'), inner = 10)
    re = randn(rng, 12)[repeat(1:12, inner = 10)]
    yg = 1 .+ 0.5x .+ re .+ exp.(0.3 .* x) .* randn(rng, n)
    yp = [Int(floor(2 * exp(0.3x[i] + 0.3re[i]) * rand(rng) * 2)) for i in 1:n]
    y1 = 1 .+ 0.5x .+ randn(rng, n); y2 = 0.3 .- 0.4x2 .+ randn(rng, n)
    hole = [5, 40, 77, 100]
    full(v, T) = Vector{Union{Missing,T}}(v)
    holed(v, T) = (w = full(v, T); w[hole] .= missing; w)
    d(case, miss) = case == :gauss ? (; y = miss ? holed(yg, Float64) : full(yg, Float64), x, g) :
                    case == :pois ? (; y = miss ? holed(yp, Int) : full(yp, Int), x, g) :
                    (; y1 = miss ? holed(y1, Float64) : full(y1, Float64),
                       y2 = miss ? holed(y2, Float64) : full(y2, Float64), x, x2)
    cases = (
        ("Gaussian sigma ~ x", () -> bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(), :gauss),
        ("Poisson", () -> bf(@formula(y ~ x)), Poisson(), :pois),
        ("Gaussian random intercept", () -> bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(), :gauss),
        ("bivariate Gaussian", () -> bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x2),
            sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1)),
            Gaussian(), :biv))
    for (name, mk, fam, c) in cases, (first_miss, second_miss) in ((false, true), (true, false))
        @testset "$name, first fit missing=$first_miss then missing=$second_miss" begin
            f = mk()
            _reuse_quiet(() -> drm(f, fam; data = d(c, first_miss)))
            reused = _reuse_quiet(() -> drm(f, fam; data = d(c, second_miss)))
            fresh = _reuse_quiet(() -> drm(mk(), fam; data = d(c, second_miss)))
            @test _reuse_same(reused, fresh)
        end
    end

    @testset "factor levels follow each data set" begin
        lv(labs) = repeat(labs, inner = n ÷ length(labs))
        dA = (; y = yg, x, f = lv(["a", "b", "c"])); dB = (; y = yg, x, f = lv(["u", "v", "w", "z"]))
        mk = () -> bf(@formula(y ~ x + f), @formula(sigma ~ x))
        for (d1, d2) in ((dA, dB), (dB, dA))
            f = mk()
            _reuse_quiet(() -> drm(f, Gaussian(); data = d1))
            @test _reuse_same(_reuse_quiet(() -> drm(f, Gaussian(); data = d2)),
                            _reuse_quiet(() -> drm(mk(), Gaussian(); data = d2)))
        end
    end
end

# The second half of the fix, which a reuse test cannot see: a response typed
# Vector{Union{Missing,Float64}} with no missing value, fitted from a FRESH formula object,
# must still allow `predict(fit, newdata)`. Without the `Missing <: eltype(raw_response)`
# clause in `_design` the stored schema treats the response as categorical and `predict` fails
# with "could not build the `mu` design matrix" (KeyError: key 0.0).
@testset "predict(newdata) after a fit on a Union{Missing,Float64} response (fresh formula)" begin
    rng = Xoshiro(32); n = 80
    x = randn(rng, n); y = 1 .+ 0.5x .+ exp.(0.3 .* x) .* randn(rng, n)
    nd = (; x = [0.0, 1.0])
    fu = _reuse_quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian();
                                data = (; y = Vector{Union{Missing,Float64}}(y), x)))
    ff = _reuse_quiet(() -> drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x)))
    pu = predict(fu, nd)
    @test all(isfinite, pu)
    @test pu ≈ predict(ff, nd)
end
