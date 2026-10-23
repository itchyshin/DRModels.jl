# Red-proof for sweep-oct16 t6: simulate() has no newdata (twin of drmTMB #1536).
# After the fix, `simulate(fit; newdata = nd)` draws one row per nd row at the
# population-level predicted parameters (RE = 0), matching predict().
using DRModels
using Test, Random
import Distributions

@testset "FX2 simulate(newdata)" begin
    Random.seed!(20261016)
    n = 40
    x = randn(n)
    y = 0.2 .+ 0.5 .* x .+ 0.4 .* randn(n)
    data = (; y, x)
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data)

    @testset "newdata formal exists and draws nd rows" begin
        nd = (x = [-1.0, 0.0, 1.0],)
        ys = simulate(fit; newdata = nd, nsim = 1, rng = MersenneTwister(1))
        @test ys isa AbstractVector
        @test length(ys) == 3
        Y = simulate(fit; newdata = nd, nsim = 5, rng = MersenneTwister(2))
        @test Y isa AbstractMatrix
        @test size(Y) == (3, 5)
        @test Y[:, 1] != Y[:, 2]
    end

    @testset "omitting newdata is still training rows (back-compat)" begin
        ys = simulate(fit; nsim = 1, rng = MersenneTwister(3))
        @test length(ys) == n
    end

    @testset "sibling families: Poisson and LogNormal newdata length" begin
        yp = Float64[rand(Distributions.Poisson(exp(0.1 + 0.4 * xi))) for xi in x]
        fp = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = yp, x))
        ys = simulate(fp; newdata = (x = [0.0, 1.0],), rng = MersenneTwister(4))
        @test length(ys) == 2
        @test all(>=(0), ys)

        yl = exp.(0.2 .+ 0.3 .* x .+ 0.25 .* randn(n))
        fl = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = (; y = yl, x))
        ys = simulate(fl; newdata = (x = [-0.5, 0.5],), rng = MersenneTwister(5))
        @test length(ys) == 2
        @test all(>(0), ys)
    end

    @testset "bivariate Gaussian newdata" begin
        y2 = -0.1 .+ 0.3 .* x .+ 0.4 .* randn(n)
        bv = drm(bf(mu1 = @formula(y ~ x), mu2 = @formula(y2 ~ x),
                    sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                    rho12 = @formula(rho12 ~ 1)), Gaussian(); data = (; y, y2, x))
        s = simulate(bv; newdata = (x = [0.0, 1.0],), rng = MersenneTwister(6))
        @test haskey(s, :mu1) && haskey(s, :mu2)
        @test length(s[:mu1]) == 2
        @test length(s[:mu2]) == 2
    end
end
