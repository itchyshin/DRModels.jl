# Binomial `residuals(fit; type = :quantile)` on two routes that used to throw:
#   1. a fit with missing responses (InexactError: the dropped rows carry a NaN
#      proportion that was rounded to an integer count), and
#   2. `(1 | g) + (0 + x | g)` and crossed `(1 | g) + (1 | h)` (KeyError :trials:
#      those routes kept no trial counts).
# The first follows the Poisson / Gamma convention: one entry per data row, NaN at
# the rows dropped for a missing response. Values are checked against an
# independent randomized-PIT computation with the same RNG stream.
using DRModels
using Test, Random, Statistics
import Distributions

# r_i = Φ⁻¹(u_i), u_i = F(y_i − 1) + (F(y_i) − F(y_i − 1)) U_i, F the Binomial(n_i, p_i)
# CDF, U_i from `rng` (one draw per row, in row order, missing rows included).
function hand_binomial_qr(p, ntr, s, rng)
    r = Vector{Float64}(undef, length(p))
    for i in eachindex(p)
        U = rand(rng)
        if ismissing(s[i])
            r[i] = NaN
            continue
        end
        d = Distributions.Binomial(Int(ntr[i]), clamp(p[i], eps(), 1 - eps()))
        a = Distributions.cdf(d, Int(s[i]) - 1); b = Distributions.cdf(d, Int(s[i]))
        r[i] = Distributions.quantile(Distributions.Normal(), clamp(a + (b - a) * U, eps(), 1 - eps()))
    end
    return r
end

same(a, b) = length(a) == length(b) && all(i -> (isnan(a[i]) && isnan(b[i])) || a[i] ≈ b[i], eachindex(a))

n = 80
x = [sin(i) for i in 1:n]
g = repeat(string.("g", 1:8), inner = 10)
yb = [sin(3i) + 0.8x[i] > 0.1 ? 1 : 0 for i in 1:n]
miss = [i % 9 == 0 for i in 1:n]

@testset "Binomial quantile residuals: missing responses" begin
    ym = Union{Missing,Int}[miss[i] ? missing : yb[i] for i in 1:n]
    fit = drm(bf(@formula(y ~ x)), Binomial(); data = (; y = ym, x))
    r = residuals(fit; type = :quantile, rng = MersenneTwister(11))
    @test length(r) == n                          # one entry per data row, as Poisson and Gamma
    @test findall(isnan, r) == findall(miss)
    @test all(isfinite, r[.!miss])
    @test same(r, hand_binomial_qr(fitted(fit), ones(n), ym, MersenneTwister(11)))
    @test length(r) == length(residuals(fit))     # same shape as the response residuals

    # grouped trials with missing rows
    ntr = [2 + i % 5 for i in 1:n]
    sg = [min(ntr[i], round(Int, ntr[i] * (0.5 + 0.4 * sin(2i) * (x[i] > 0 ? 1 : -1)))) for i in 1:n]
    sm = Union{Missing,Float64}[miss[i] ? missing : Float64(sg[i]) for i in 1:n]
    fm = Union{Missing,Float64}[miss[i] ? missing : Float64(ntr[i] - sg[i]) for i in 1:n]
    fitg = drm(bf(@formula(cbind(s, f) ~ x)), Binomial(); data = (; s = sm, f = fm, x))
    rg = residuals(fitg; type = :quantile, rng = MersenneTwister(5))
    @test length(rg) == n && findall(isnan, rg) == findall(miss)
    @test same(rg, hand_binomial_qr(fitted(fitg), ntr, sm, MersenneTwister(5)))

    # a random intercept on the mean with missing rows (σ_b-marginal route)
    fitr = drm(bf(@formula(y ~ x + (1 | g))), Binomial(); data = (; y = ym, x, g))
    rr = residuals(fitr; type = :quantile, rng = MersenneTwister(3))
    @test length(rr) == n && findall(isnan, rr) == findall(miss)

    # BetaBinomial shares the discrete driver
    fitb = drm(bf(@formula(cbind(s, f) ~ x)), BetaBinomial(); data = (; s = sm, f = fm, x))
    rb = residuals(fitb; type = :quantile, rng = MersenneTwister(4))
    @test length(rb) == n && findall(isnan, rb) == findall(miss)
end

@testset "Binomial quantile residuals: crossed (1 | g) + (1 | h)" begin
    h = string.("h", [1 + (7i % 4) for i in 1:n])
    ntr = [2 + i % 5 for i in 1:n]
    sg = [min(ntr[i], round(Int, ntr[i] * (0.5 + 0.4 * sin(2i) * (x[i] > 0 ? 1 : -1)))) for i in 1:n]
    data = (; s = Float64.(sg), f = Float64.(ntr .- sg), x, g, h)
    for kw in ((;), (; marginal = :AGHQ))
        fit = drm(bf(@formula(cbind(s, f) ~ x + (1 | g) + (1 | h))), Binomial(); data = data, kw...)
        @test fit.scales[:trials] == Float64.(ntr)
        r = residuals(fit; type = :quantile, rng = MersenneTwister(6))
        @test length(r) == n && all(isfinite, r)
        @test same(r, hand_binomial_qr(fitted(fit), ntr, sg, MersenneTwister(6)))
    end
end

@testset "Binomial quantile residuals: (1 | g) + (0 + x | g), Laplace" begin
    fit = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g))), Binomial();
              data = (; y = yb, x, g), marginal = :Laplace)
    @test haskey(fit.scales, :trials) && fit.scales[:trials] == ones(n)
    r = residuals(fit; type = :quantile, rng = MersenneTwister(2))
    @test length(r) == n && all(isfinite, r)
    # random effects set to 0 on this route (documented for crossed / slope shapes)
    @test same(r, hand_binomial_qr(fitted(fit), ones(n), yb, MersenneTwister(2)))

    # grouped trials: the stored counts are the real ones, not ones
    ntr = [2 + i % 5 for i in 1:n]
    sg = [min(ntr[i], round(Int, ntr[i] * (0.5 + 0.4 * sin(2i) * (x[i] > 0 ? 1 : -1)))) for i in 1:n]
    fitg = drm(bf(@formula(cbind(s, f) ~ x + (1 | g) + (0 + x | g))), Binomial();
               data = (; s = Float64.(sg), f = Float64.(ntr .- sg), x, g), marginal = :Laplace)
    @test fitg.scales[:trials] == Float64.(ntr)
    rg = residuals(fitg; type = :quantile, rng = MersenneTwister(9))
    @test length(rg) == n && all(isfinite, rg)
    @test same(rg, hand_binomial_qr(fitted(fitg), ntr, sg, MersenneTwister(9)))
end
