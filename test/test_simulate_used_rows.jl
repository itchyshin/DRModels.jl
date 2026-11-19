using DRModels, Test, Random

# #1352: `simulate(fit)` returns the rows used in the fit (`nobs`), as drmTMB does; a row dropped
# for a missing response is not simulated. Deterministic data, the issue's reproducer.
@testset "#1352 simulate returns the used rows" begin
    n = 60
    x = [sin(i) for i in 1:n]
    c = Union{Missing,Int}[(i % 6 == 0) ? missing : round(Int, 3 + 2x[i] + 0.5cos(7i)) for i in 1:n]
    keep = [!ismissing(v) for v in c]
    f = drm(bf(@formula(c ~ x)), Poisson(); data = (; c, x))
    @test nobs(f) == 50 == count(keep)
    @test length(fitted(f)) == nobs(f)
    @test length(simulate(f; rng = MersenneTwister(1))) == nobs(f)
    @test size(simulate(f; nsim = 3, rng = MersenneTwister(1))) == (nobs(f), 3)
    @test size(simulate(f; nsim = 2), 1) == nobs(f)
    # the draw at the used rows is the full-design draw cut to those rows (same seed)
    all_rows = DRModels._simulate_all_rows(f; rng = MersenneTwister(7))
    @test length(all_rows) == n
    @test simulate(f; rng = MersenneTwister(7)) == all_rows[keep]
    M = DRModels._simulate_all_rows(f; nsim = 3, rng = MersenneTwister(7))
    @test simulate(f; nsim = 3, rng = MersenneTwister(7)) == M[keep, :]
    # `newdata` is unchanged: one row per newdata row
    @test size(simulate(f; newdata = (x = [0.0, 1.0, 2.0],), nsim = 2, rng = MersenneTwister(1))) == (3, 2)

    # Gaussian with a masked (NaN) response
    y = 1.0 .+ 0.5 .* x .+ 0.2 .* cos.(3 .* (1:n))
    ym = copy(y); ym[[3, 10, 11]] .= NaN
    fg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (y = ym, x = x))
    @test nobs(fg) == n - 3
    @test length(simulate(fg; rng = MersenneTwister(2))) == nobs(fg)
    @test size(simulate(fg; nsim = 4, rng = MersenneTwister(2))) == (nobs(fg), 4)

    # no missing response: nothing changes
    fc = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (y = y, x = x))
    @test length(simulate(fc; rng = MersenneTwister(2))) == n
    @test simulate(fc; rng = MersenneTwister(2)) == DRModels._simulate_all_rows(fc; rng = MersenneTwister(2))
end
