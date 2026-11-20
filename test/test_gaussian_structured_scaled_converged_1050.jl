# DRModels.jl#1050 (cluster B): the structured Gaussian fitters and the
# random-intercept routes of gaussian_ranef.jl read `converged` from the unit-free
# Newton-step flag (PR #1044), not `Optim.g_converged` (absolute 1e-8). The same
# stationary fit with a predictor rescaled by 1e4 must read converged, and a
# bootstrap of it must not drop a replicate.
using DRModels
using Test
using Random
using LinearAlgebra

@testset "#1050 cluster B: scaled predictor stays converged" begin
    rng = MersenneTwister(7)
    G = 40; m = 6; n = G * m
    id = repeat(1:G, inner = m)
    x0 = randn(rng, n)
    A = let M = randn(rng, G, G); M * M' / G + I end
    d = sqrt.(diag(A)); K = A ./ (d * d')
    u = 0.8 .* (cholesky(Symmetric(K)).L * randn(rng, G))
    b = 0.4 .* randn(rng, G)
    h = repeat(1:8, outer = n ÷ 8)
    y = 0.3 .+ 0.5 .* x0 .+ u[id] .+ b[id] .+ exp.(-0.3 .+ 0.1 .* x0) .* randn(rng, n)

    for sc in (1.0, 1e4)
        data = (; y, x = x0 .* sc, id, g = id, h)

        # gaussian_structured.jl: structured intercept, varying scale.
        f1 = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ x)), Gaussian();
                 data, K = K)
        @test is_converged(f1)

        # structured + ordinary bar: failed at scale 1e4 before the switch.
        f2 = drm(bf(@formula(y ~ x + relmat(1 | id) + (1 | g)), @formula(sigma ~ x)),
                 Gaussian(); data = (; y, x = x0 .* sc, id, g = h), K = K)
        @test is_converged(f2)

        # gaussian_ranef.jl: random intercept on the log-sigma axis.
        f3 = drm(bf(@formula(y ~ x), @formula(sigma ~ x + (1 | g))), Gaussian(); data)
        @test is_converged(f3)

        # gaussian_ranef.jl: two ordinary random intercepts.
        f4 = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ x)), Gaussian();
                 data)
        @test is_converged(f4)
    end

    # The scaled-predictor replicate must keep every bootstrap draw.
    form = bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ x))
    dat = (; y, x = x0 .* 1e4, id)
    res = bootstrap_result(form, Gaussian(); data = dat, K = K, B = 6,
                           rng = MersenneTwister(1050))
    @test res.failed == 0
    @test res.used == 6
    @test res.status == "bootstrap"
end
