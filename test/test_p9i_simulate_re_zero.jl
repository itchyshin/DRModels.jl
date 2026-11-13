# P9i: `simulate(fit; re = :zero)` must draw with every random effect set to zero.
# The older sd-formula engines stored the CONDITIONAL sigma (and, for the Gaussian joint mean
# and sigma route, the conditional mean), so a "zero random effect" draw still carried the
# fitted group modes: the draw SD was 0.846 against a population sigma of 0.698 in the
# review that confirmed it (review-g6e-sigma-re.md).
using DRModels
using Test, Random
import Distributions

function _p9i_data(; seed = 9, G = 40, m = 15)
    rng = Random.Xoshiro(seed)
    n = G * m
    g = repeat(1:G, inner = m)
    x = randn(rng, n)
    wg = collect(range(-1, 1; length = G))
    w = wg[g]
    b = 0.5 .* randn(rng, G)             # mean random effect
    bs = 0.5 .* randn(rng, G)            # sigma random effect (log scale)
    μ = exp.(0.5 .+ 0.1 .* x)
    ygau = 0.3 .+ 0.5 .* x .+ b[g] .+ exp.(-0.3 .+ bs[g]) .* randn(rng, n)
    ynb = Float64[rand(rng, Distributions.NegativeBinomial(exp(-2 * (-0.4 + bs[g[i]])),
                  exp(-2 * (-0.4 + bs[g[i]])) / (exp(-2 * (-0.4 + bs[g[i]])) + μ[i]))) for i in 1:n]
    ygam = Float64[rand(rng, Distributions.Gamma(exp(-2 * (-0.7 + bs[g[i]])),
                  μ[i] * exp(2 * (-0.7 + bs[g[i]])))) for i in 1:n]
    return (; ygau, ynb, ygam, x, g, w)
end

# Mean over rows of the across-draw sample variance of each row.
_p9i_rowvar(S) = sum(var(view(S, i, :)) for i in 1:size(S, 1)) / size(S, 1)
using Statistics: var

@testset "simulate(re = :zero) carries no random-effect variation (P9i)" begin
    d = _p9i_data()
    n = length(d.x)

    @testset "Gaussian sigma random effect with an sd() formula" begin
        fit = drm(bf(@formula(ygau ~ x), @formula(sigma ~ 1 + (1 | g)), @formula(sd(g, sigma) ~ w)),
                  Gaussian(); data = d)
        @test fit.converged
        σpop = exp.(hcat(ones(n), zeros(n))[:, 1] .* coef(fit, :sigma)[1])
        @test fit.scales[:sigma] != σpop                     # the stored sigma is conditional
        y0 = simulate(fit; re = :zero, rng = Random.Xoshiro(1))
        @test y0 ≈ fit.means[:mu] .+ σpop .* randn(Random.Xoshiro(1), n)
    end

    @testset "Gaussian joint mean and sigma random effects" begin
        fit = drm(bf(@formula(ygau ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g)),
                     @formula(sd(g) ~ w), @formula(sd(g, sigma) ~ w)), Gaussian(); data = d)
        @test fit.converged
        βμ = coef(fit, :mu)
        μpop = βμ[1] .+ βμ[2] .* d.x
        σpop = fill(exp(coef(fit, :sigma)[1]), n)
        @test !(fit.means[:mu] ≈ μpop)                       # the stored mean is conditional
        y0 = simulate(fit; re = :zero, rng = Random.Xoshiro(2))
        @test y0 ≈ μpop .+ σpop .* randn(Random.Xoshiro(2), n)
    end

    @testset "NB2 sigma random effect: draw variance is the population one" begin
        fit = drm(bf(@formula(ynb ~ 1), @formula(sigma ~ 1 + (1 | g)), @formula(sd(g, sigma) ~ w)),
                  NegBinomial2(); data = d)
        @test fit.converged
        μ0 = exp(coef(fit, :mu)[1]); σ0 = exp(coef(fit, :sigma)[1])
        S = simulate(fit; re = :zero, nsim = 4000, rng = Random.Xoshiro(3))
        @test isapprox(_p9i_rowvar(S), μ0 + σ0^2 * μ0^2; rtol = 0.06)
        # the conditional sigma would give a clearly larger variance
        cond = sum(fit.means[:mu] .+ fit.scales[:sigma] .^ 2 .* fit.means[:mu] .^ 2) / n
        @test cond > 1.1 * (μ0 + σ0^2 * μ0^2)
    end

    @testset "Gamma sigma random effect: draw variance is the population one" begin
        fit = drm(bf(@formula(ygam ~ 1), @formula(sigma ~ 1 + (1 | g)), @formula(sd(g, sigma) ~ w)),
                  Gamma(); data = d)
        @test fit.converged
        μ0 = exp(coef(fit, :mu)[1]); σ0 = exp(coef(fit, :sigma)[1])
        S = simulate(fit; re = :zero, nsim = 4000, rng = Random.Xoshiro(4))
        @test isapprox(_p9i_rowvar(S), σ0^2 * μ0^2; rtol = 0.06)
        @test sum(fit.scales[:sigma] .^ 2 .* fit.means[:mu] .^ 2) / n > 1.1 * σ0^2 * μ0^2
    end

    @testset "fits that already stored population values are unchanged" begin
        fit = drm(bf(@formula(ygau ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ w)),
                  Gaussian(); data = d)
        y0 = simulate(fit; re = :zero, rng = Random.Xoshiro(5))
        @test y0 == fit.means[:mu] .+ fit.scales[:sigma] .* randn(Random.Xoshiro(5), n)
    end

    @testset "the bootstrap fallback keeps its conditional draw" begin
        # `_simulate_conditional` is what the bootstrap fallback calls; it is the old `re = :zero` draw.
        fit = drm(bf(@formula(ygau ~ x), @formula(sigma ~ 1 + (1 | g)), @formula(sd(g, sigma) ~ w)),
                  Gaussian(); data = d)
        y = DRModels._simulate_conditional(fit, Random.Xoshiro(6))
        @test y == fit.means[:mu] .+ fit.scales[:sigma] .* randn(Random.Xoshiro(6), n)
    end
end

# Routes that the first set of checks did not pin: BetaBinomial, the structured (relmat) sigma
# routes, and the NB2 and Gamma joint mean-and-sigma routes. For each, the reference is the
# package's own population-level path: `simulate(fit; newdata = data, re = :zero)` with the same
# seed. On main the stored values on these routes are conditional, so the two draws differ
# (gaps of several counts); after the change they are equal.
using LinearAlgebra: cholesky, Symmetric

function _p9i_data2(; seed = 19, G = 40, m = 15)
    rng = Random.Xoshiro(seed)
    n = G * m
    g = repeat(1:G, inner = m)
    x = randn(rng, n)
    wg = collect(range(-1, 1; length = G)); w = wg[g]
    b = 0.5 .* randn(rng, G); bs = 0.5 .* randn(rng, G)
    μ = exp.(0.5 .+ 0.1 .* x)
    ynb = Float64[rand(rng, Distributions.NegativeBinomial(exp(-2 * (-0.4 + bs[g[i]])),
                  exp(-2 * (-0.4 + bs[g[i]])) / (exp(-2 * (-0.4 + bs[g[i]])) + μ[i] * exp(b[g[i]])))) for i in 1:n]
    ygam = Float64[rand(rng, Distributions.Gamma(exp(-2 * (-0.7 + bs[g[i]])),
                  μ[i] * exp(b[g[i]]) * exp(2 * (-0.7 + bs[g[i]])))) for i in 1:n]
    tr = 20
    pm = 1 ./ (1 .+ exp.(-(0.1 .+ 0.4 .* x .+ b[g])))
    prec = exp.(-2 .* (-0.5 .+ bs[g]))
    s = [rand(rng, Distributions.BetaBinomial(tr, pm[i] * prec[i], (1 - pm[i]) * prec[i])) for i in 1:n]
    K = [0.3^abs(i - j) for i in 1:G, j in 1:G]
    a = 0.5 .* (cholesky(Symmetric(K)).L * randn(rng, G))
    yst = 0.3 .+ 0.5 .* x .+ exp.(-0.3 .+ a[g]) .* randn(rng, n)
    nbst = Float64[rand(rng, Distributions.NegativeBinomial(exp(-2 * (-0.4 + a[g[i]])),
                   exp(-2 * (-0.4 + a[g[i]])) / (exp(-2 * (-0.4 + a[g[i]])) + μ[i]))) for i in 1:n]
    ids = [string('s', lpad(string(gi), 2, '0')) for gi in g]
    return (; ynb, ygam, succ = Float64.(s), fail = Float64.(tr .- s), x, g, w, id = ids, yst, nbst), K
end

@testset "re = :zero equals the population-level draw on the remaining routes (P9i)" begin
    d, K = _p9i_data2()
    cases = [
        "BetaBinomial sd(g, sigma)" => () -> drm(bf(@formula(cbind(succ, fail) ~ x),
            @formula(sigma ~ 1 + (1 | g)), @formula(sd(g, sigma) ~ w)), BetaBinomial(); data = d, se = false),
        "Gaussian relmat sigma" => () -> drm(bf(@formula(yst ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
            @formula(sd(id, relmat, sigma) ~ 1)), Gaussian(); data = d, K = K),
        "Gaussian relmat sigma, sd ~ w" => () -> drm(bf(@formula(yst ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
            @formula(sd(id, relmat, sigma) ~ w)), Gaussian(); data = d, K = K),
        "NB2 relmat sigma" => () -> drm(bf(@formula(nbst ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
            @formula(sd(id, relmat, sigma) ~ 1)), NegBinomial2(); data = d, K = K),
        "Gamma relmat sigma" => () -> drm(bf(@formula(ygam ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
            @formula(sd(id, relmat, sigma) ~ 1)), Gamma(); data = d, K = K),
        "NB2 joint mean and sigma" => () -> drm(bf(@formula(ynb ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g)),
            @formula(sd(g) ~ w), @formula(sd(g, sigma) ~ w)), NegBinomial2(); data = d),
        "Gamma joint mean and sigma" => () -> drm(bf(@formula(ygam ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g)),
            @formula(sd(g) ~ w), @formula(sd(g, sigma) ~ w)), Gamma(); data = d),
    ]
    for (label, mk) in cases
        @testset "$label" begin
            fit = mk()
            @test fit.converged
            z = simulate(fit; re = :zero, rng = Random.Xoshiro(11))
            p = simulate(fit; newdata = d, re = :zero, rng = Random.Xoshiro(11))
            @test length(z) == length(p)
            @test maximum(abs.(vec(z) .- vec(p))) <= 1e-8 * max(1.0, maximum(abs.(vec(p))))
        end
    end
end
