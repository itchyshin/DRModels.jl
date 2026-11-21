# End-to-end coupled `simulate(...; re = :zero)` in src/locscale_simulate.jl.
using DRModels
using Test, Random, Statistics

@testset "locscale_simulate: marginal draws on coupled Gamma fit" begin
    G = 4
    pair = [40, 10, 30, 20, 10, 40, 20, 30, 30, 20, 40, 10, 20, 30, 10, 40,
            40, 20, 30, 10, 10, 30, 20, 40]
    g = [40, 10, 30, 20, 10, 40, 20, 30, 30, 20, 40, 10, 20, 30, 10, 40,
         40, 20, 30, 10, 10, 30, 20, 40]
    n = length(g)
    x = repeat([-0.8, -0.3, 0.2, 0.7], 6)
    y = [1.2 + 0.05 * i for i in 1:n]
    fit = drm(bf(@formula(y ~ x + (1 | pair | g)), @formula(sigma ~ x + (1 | pair | g))),
              Gamma(); data = (; y, x, g, pair), se = false)
    @test isfinite(loglik(fit))
    @test fit.nll isa DRModels.LocScaleObjective

    Y = simulate(fit; nsim = 120, rng = MersenneTwister(11), re = :zero)
    @test size(Y) == (nobs(fit), 120)
    @test all(isfinite, Y)
    @test all(>(0), Y)
    μpop = fitted(fit; re = :zero)
    @test mean(vec(mean(Y; dims = 1))) ≈ mean(μpop) rtol = 0.2
end