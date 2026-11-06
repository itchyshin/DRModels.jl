# #1119: bootstrap_ci keeps the plain percentile interval, but warns that percentile
# intervals for sigma terms in a mixed model under-cover and points to profile intervals.
using DRModels
using Test, Random, Logging

@testset "bootstrap_ci warns on sigma terms in a mixed model (#1119)" begin
    rng = MersenneTwister(20261016902)
    G, n_g = 10, 10
    g = repeat(1:G, inner = n_g)
    x = repeat(collect(range(-1, 1, length = n_g)), G)
    u = 0.1 .* randn(rng, G)
    y = 0.5 .* x .+ u[g] .+ exp.(0.3 .* x) .* randn(rng, G * n_g)
    data = (; y, x, g)
    form = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1 + x))
    fit = drm(form, Gaussian(); data)

    # fit-based and formula-based forms both warn, and name the profile route
    @test_logs (:warn, r"profile") match_mode = :any bootstrap_ci(
        fit; data, B = 3, rng = MersenneTwister(1), failures = :skip)
    @test_logs (:warn, r"profile") match_mode = :any bootstrap_ci(
        form, Gaussian(); data, B = 3, rng = MersenneTwister(1), failures = :skip)

    # the interval itself is still the plain percentile interval (same rows as bootstrap_summary)
    ci = Logging.with_logger(Logging.NullLogger()) do
        bootstrap_ci(fit; data, B = 3, rng = MersenneTwister(1), failures = :skip)
    end
    bs = Logging.with_logger(Logging.NullLogger()) do
        bootstrap_summary(fit; data, B = 3, rng = MersenneTwister(1), failures = :skip)
    end
    @test [(r.param, r.coef, r.lower, r.upper) for r in ci] ==
          [(r.param, r.coef, r.lower, r.upper) for r in bs]

    # no warning without a random effect
    yf = 0.5 .* x .+ exp.(0.3 .* x) .* randn(rng, G * n_g)
    dataf = (; y = yf, x, g)
    formf = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x))
    @test_logs bootstrap_ci(formf, Gaussian(); data = dataf, B = 3, rng = MersenneTwister(1))
end
