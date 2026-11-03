# test/test_resd_profile_boundary.jl
# Issue #1117: ordinary (1|g) profile CI for sd_g at the SD floor must never
# throw and must never return a successful (-Inf, Inf). drmTMB's endpoint
# engine treats a lower arm that reaches the log-SD floor without crossing
# the LR threshold as a boundary and reports the SD-scale interval [0, u].
using Test
using DRModels
using Random

function _p1117_fit(spec_seed)
    rng = MersenneTwister(spec_seed)
    G = 10
    n_g = 10
    sd_g = 0.1
    g = repeat(1:G, inner = n_g)
    x = repeat(collect(range(-1, 1, length = n_g)), G)
    u = sd_g .* randn(rng, G)
    sigma = exp.(0.0 .+ 0.3 .* x)
    y = 0.0 .+ 0.5 .* x .+ u[g] .+ sigma .* randn(rng, length(x))
    return drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1 + x)),
               Gaussian(); data = (; y, x, g))
end

@testset "#1117 resd profile at the SD floor is [0, u], never throws" begin
    # Seed 20261016900: pre-fix, profile_result stored (-Inf, finite] with
    # lower_reason = not_converged and confint threw ArgumentError.
    # Seed 20261016902: pre-fix, log σ_b ≈ -20 and confint returned (-Inf, Inf)
    # with failed = 0 (treated as a successful unbounded interval).
    for spec_seed in (20261016900, 20261016902)
        fit = _p1117_fit(spec_seed)
        pr = profile_result(fit; parm = :resd)
        row = only(pr.ci)
        stat = only(pr.stats)
        @test row.param === :resd
        # User-facing interval: SD-scale [0, u] with a finite upper.
        @test row.lower == 0.0
        @test isfinite(row.upper) && row.upper > 0.0
        @test row.lower < row.upper
        @test !stat.lower_endpoint_failed
        @test !stat.upper_endpoint_failed
        @test pr.failed == 0
        ci = confint(fit; method = :profile, parm = :resd)
        crow = only(ci)
        @test crow.lower == 0.0
        @test isfinite(crow.upper) && crow.upper > 0.0
        @test !isinf(crow.lower) && !isinf(crow.upper)
        # True sd_g = 0.1 must sit inside the boundary interval.
        @test crow.lower <= 0.1 <= crow.upper
    end

    # Sibling path: a well-identified crossed (1|g)+(1|h) SD stays on the
    # working (log σ_b) scale. The [0, u] rewrite is only for a floor/boundary
    # lower arm, not every :resd row.
    rng = MersenneTwister(20260614)
    G = 7; H = 6; n = 260
    x = randn(rng, n)
    gid = rand(rng, 1:G, n); hid = rand(rng, 1:H, n)
    g = Symbol.("g", gid); h = Symbol.("h", hid)
    bg = 0.45 .* randn(rng, G); bh = 0.35 .* randn(rng, H)
    y = [0.25 + 0.55 * x[i] + bg[gid[i]] + bh[hid[i]] + 0.45 * randn(rng) for i in 1:n]
    crossed = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                  Gaussian(); data = (; y, x, g, h))
    resd = confint(crossed; method = :profile, parm = :resd)
    @test all(r -> r.lower != 0.0, resd)  # not rewritten to the SD-scale boundary form
    @test all(r -> isfinite(r.lower) && isfinite(r.upper), resd)
    @test all(r -> r.lower < r.estimate < r.upper, resd)
end
