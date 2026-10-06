# Parametric simulation from a fitted Gaussian model (the building block for a
# parametric bootstrap). Residual-level draws: y* = μ̂ + (model residual SD)·z.
using DRModels
using Test, Random
import Distributions

@testset "simulate" begin
    Random.seed!(9)
    n = 600
    x = randn(n)
    y = 0.5 .+ 0.4 .* x .+ exp.(-0.2 .+ 0.3 .* x) .* randn(n)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))

    ys = simulate(fit; rng = MersenneTwister(1))
    @test length(ys) == n
    @test ys != y                                   # a fresh draw, not the data

    # refit on simulated data recovers similar coefficients (bootstrap sanity)
    fit2 = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y = ys, x))
    @test coef(fit2, :mu) ≈ coef(fit, :mu) atol = 0.3

    # bivariate: per-response draws
    y2 = -0.3 .+ 0.5 .* x .+ 0.4 .* randn(n)
    bv = drm(bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
                sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                rho12 = @formula(rho12 ~ 1)), Gaussian(); data = (; y, y2, x))
    s = simulate(bv; rng = MersenneTwister(2))
    @test haskey(s, :mu1) && haskey(s, :mu2)
    @test length(s[:mu1]) == n

    # nsim contract: Vector for nsim==1, nobs × nsim Matrix for nsim>1
    @test simulate(fit; nsim = 1, rng = MersenneTwister(3)) isa AbstractVector
    Y = simulate(fit; nsim = 5, rng = MersenneTwister(3))
    @test Y isa AbstractMatrix
    @test size(Y) == (n, 5)
    @test Y[:, 1] != Y[:, 2]                          # independent columns
    @test_throws ArgumentError simulate(fit; nsim = 0)

    # bivariate nsim>1: a length-nsim Vector of Dict draws
    sv = simulate(bv; nsim = 3, rng = MersenneTwister(4))
    @test sv isa AbstractVector && length(sv) == 3
    @test all(d -> haskey(d, :mu1) && haskey(d, :mu2), sv)
end

# Poisson: simulate counts from the fitted mean and recover the slope on refit.
@testset "simulate Poisson" begin
    Random.seed!(11)
    n = 800
    x = randn(n)
    yp = Float64[rand(Distributions.Poisson(exp(0.2 + 0.5 * xi))) for xi in x]
    fit = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = yp, x))

    ys = simulate(fit; rng = MersenneTwister(7))
    @test length(ys) == n
    @test all(>=(0), ys)                              # counts are non-negative
    @test all(v -> v == round(v), ys)                 # integer-valued draws
    @test ys != yp

    Y = simulate(fit; nsim = 4, rng = MersenneTwister(7))
    @test size(Y) == (n, 4)

    fit2 = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = ys, x))
    @test coef(fit2, :mu) ≈ coef(fit, :mu) atol = 0.25
end
