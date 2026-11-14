# Zero-anchored profile interval for a variance component whose estimate is at zero:
# lower limit exactly 0, upper limit from the profile deviance crossing at the
# chi-square(1) quantile (the cut-off `confint(...; method = :profile)` already uses).
using DRModels, Test, Random, Logging
import Distributions as D

# Hand-written profile deviance for y = X beta + b_g + e, balanced groups, one random
# intercept SD sb and residual SD se: beta by GLS, se by golden-section search.
function _hand_loglik(y, X, G, m, sb, se)
    v = se^2; w = sb^2 / (v + m * sb^2)
    A = zeros(size(X, 2), size(X, 2)); c = zeros(size(X, 2))
    for i in 1:G
        r = (i - 1) * m + 1:i * m
        Xi = X[r, :]; yi = y[r]
        Vi_inv(z) = (z .- w * sum(z)) ./ v
        A .+= Xi' * reduce(hcat, Vi_inv(Xi[:, j]) for j in 1:size(X, 2))
        c .+= Xi' * Vi_inv(yi)
    end
    β = A \ c
    q = 0.0
    for i in 1:G
        r = (i - 1) * m + 1:i * m
        e = y[r] .- X[r, :] * β
        q += sum(abs2, e) / v - w * sum(e)^2 / v
    end
    n = G * m
    return -0.5 * (n * log(2π) + G * ((m - 1) * log(v) + log(v + m * sb^2)) + q)
end

function _hand_profile(y, X, G, m, sb)
    f(ls) = -_hand_loglik(y, X, G, m, sb, exp(ls))
    a, b = log(0.05), log(3.0); φ = (sqrt(5) - 1) / 2
    c = b - φ * (b - a); d = a + φ * (b - a)
    for _ in 1:90
        if f(c) < f(d); b = d else a = c end
        c = b - φ * (b - a); d = a + φ * (b - a)
    end
    return _hand_loglik(y, X, G, m, sb, exp((a + b) / 2))
end

function _hand_upper(y, X, G, m, llhat; level = 0.95)
    cut = D.quantile(D.Chisq(1), level)
    lo, hi = 1e-6, 3.0
    for _ in 1:80
        mid = (lo + hi) / 2
        2 * (llhat - _hand_profile(y, X, G, m, mid)) < cut ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

function _null_data(seed; G = 20, m = 10)
    Random.seed!(seed)
    n = G * m; g = repeat(1:G, inner = m)
    x = randn(n); y = 0.4 .- 0.5 .* x .+ 0.7 .* randn(n)     # no random-effect variance
    return (; y, x, g), G, m
end

quiet(f) = with_logger(f, NullLogger())

@testset "zero-anchored profile interval" begin
    @testset "one (1 | g) at zero matches a hand-written profile" begin
        d, G, m = _null_data(202611141 * 10 + 2)
        fit = quiet() do
            drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
        end
        X = hcat(ones(G * m), d.x)
        θ = fit.theta
        @test θ[4] < log(1e-4)                         # the SD estimate is at zero
        ll0 = _hand_profile(d.y, X, G, m, 1e-9)
        @test ll0 ≈ loglikelihood(fit) atol = 1e-6
        row = only(quiet() do; confint(fit; method = :profile, parm = :resd); end)
        @test row.lower == 0.0
        @test row.upper ≈ _hand_upper(d.y, X, G, m, ll0) atol = 1e-4
    end

    @testset "independent (1 | g) + (0 + x | g), both SDs at zero" begin
        d, G, m = _null_data(777 + 2)
        fit = quiet() do
            drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1)),
                Gaussian(); data = d)
        end
        @test all(fit.theta[4:5] .< log(1e-4))
        pr = quiet() do; profile_result(fit; parm = :resd); end
        @test pr.failed == 0
        for row in pr.ci
            @test row.lower == 0.0
            @test isfinite(row.upper) && 0.05 < row.upper < 1.0
        end
        ci = quiet() do; confint(fit; method = :profile, parm = :resd); end
        @test all(r -> r.lower == 0.0 && isfinite(r.upper), ci)
    end
end
