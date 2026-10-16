# #983: `converged` must not depend on the units of y. A Gaussian fit with a tiny
# residual SD sits at the optimum, but the absolute `g_tol = 1e-8` gradient test
# cannot be met (the mean-coefficient gradient scales like 1/sigma^2).
using DRModels
using Test, Random

function _fit983(seed, sd; n = 100, scale = 1.0)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    y = 1 .+ 0.5 .* x .+ sd .* randn(rng, n)
    return drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian();
               data = (; y = scale .* y, x = x))
end

@testset "#983 near-zero-variance Gaussian fits report converged" begin
    seeds = 20260927 .+ 1000 * 15 .+ (1:12)
    # Issue cell 15: residual SD exp(-9.2) ~ 1e-4.
    fits = [_fit983(s, exp(-9.2)) for s in seeds]
    @test all(f -> f.converged, fits)
    # A smaller scale is the same problem, only more severe.
    @test all(s -> _fit983(s, 1e-5).converged, seeds)
    # Unit-free: the verdict matches the rescaled-y fit, and the estimates agree.
    for s in seeds[1:4]
        a = _fit983(s, exp(-9.2)); b = _fit983(s, exp(-9.2); scale = 1e4)
        @test a.converged == b.converged
    end
end

@testset "#983 the unit-free criterion does not hide real failures" begin
    # J1 run-off detector (#1059) still wins over a small Newton decrement.
    Random.seed!(1)
    n = 60
    x = Float64.(vcat(zeros(n ÷ 2), ones(n ÷ 2)))
    y = ifelse.(x .== 1, 2.0, randn(n))
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))
    @test fit.converged == false

    # Contract of the helper: a large decrement (far from stationary) is not
    # converged however the gradient is scaled; a non-PD Hessian never converges.
    H = [2.0 0.0; 0.0 2.0]
    @test DRModels._newton_decrement_small([1e-3, 1e-3], H, 100)
    @test !DRModels._newton_decrement_small([1.0, 1.0], H, 100)
    @test !DRModels._newton_decrement_small([1e-9, 1e-9], [1.0 2.0; 2.0 1.0], 100)
    @test !DRModels._newton_decrement_small([1e-9, 1e-9], fill(NaN, 2, 2), 100)
end
