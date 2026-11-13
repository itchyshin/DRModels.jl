# sigma(fit) must not return per-row context kept in `fit.scales` (trials,
# cumulative-logit linear predictor and cut points). A family without a fitted
# scale returns an empty Dict, as the sigma() docstring says.
using DRModels
using Test, Random

@testset "sigma() hides per-row context stored in fit.scales" begin
    rng = Random.Xoshiro(20261112)
    n = 120
    x = [sin(0.37i) + 0.3cos(1.7i) for i in 1:n]
    g = [string("g", mod(i, 6) + 1) for i in 1:n]
    ybin = [rand(rng) < 1 / (1 + exp(-(0.2 + 0.8xi))) ? 1 : 0 for xi in x]
    ntr = [5 + mod(i, 4) for i in 1:n]
    sbin = [sum(rand(rng) < 1 / (1 + exp(-(0.2 + 0.8xi))) for _ in 1:ntr[i]) for (i, xi) in enumerate(x)]
    fbin = ntr .- sbin
    sbb = [min(ntr[i], max(0, round(Int, ntr[i] / (1 + exp(-0.3x[i])) + randn(rng)))) for i in 1:n]
    fbb = ntr .- sbb
    ycum = [clamp(round(Int, 2 + 1.2xi + 0.9randn(rng)), 0, 3) + 1 for xi in x]
    d = (; x, g, ybin, sbin, fbin, sbb, fbb, ycum)

    @testset "Binomial (0/1, cbind, random intercept) -> empty Dict" begin
        for f in (drm(bf(@formula(ybin ~ x)), Binomial(); data = d),
                  drm(bf(@formula(cbind(sbin, fbin) ~ x)), Binomial(); data = d),
                  drm(bf(@formula(cbind(sbin, fbin) ~ x + (1 | g))), Binomial(); data = d))
            s = sigma(f)
            @test s isa AbstractDict
            @test isempty(s)
        end
    end

    @testset "BetaBinomial -> the sigma vector only" begin
        f = drm(bf(@formula(cbind(sbb, fbb) ~ x), @formula(sigma ~ 1)), BetaBinomial(); data = d)
        s = sigma(f)
        @test s isa Vector{Float64}
        @test length(s) == n
        @test s == f.scales[:sigma]
    end

    @testset "CumulativeLogit -> empty Dict" begin
        f = drm(bf(@formula(ycum ~ x)), CumulativeLogit(); data = d)
        @test haskey(f.scales, :ordinal_cuts)     # still stored for predict / simulate
        @test isempty(sigma(f))
    end

    @testset "Poisson and families with a scale are unchanged" begin
        yp = [round(Int, 2exp(0.4xi) * (0.5 + rand(rng))) for xi in x]
        @test isempty(sigma(drm(bf(@formula(yp ~ x)), Poisson(); data = (; yp, x))))
        yg = 0.5 .+ 0.6 .* x .+ 0.7 .* randn(rng, n)
        f = drm(bf(@formula(yg ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; yg, x))
        @test sigma(f) === f.scales[:sigma]
    end

    @testset "the stored context is still used downstream" begin
        f = drm(bf(@formula(cbind(sbin, fbin) ~ x)), Binomial(); data = d)
        @test f.scales[:trials] == Float64.(ntr)
        b = DRModels.drm_bridge(; formula = "cbind(sbin, fbin) ~ x", family = "binomial", data = d)
        @test b["sigma"] isa AbstractDict && isempty(b["sigma"])
    end
end
