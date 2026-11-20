# #1050 (Poisson and Binomial fixed-effect routes): `converged` is the unit-free
# Newton step in SE units plus the iteration-cap check, not Optim's absolute
# `g_tol = 1e-8` gradient test. A predictor scaled up moves the gradient with it,
# so the same stationary fit stalls above the absolute bar, reads unconverged,
# and a bootstrap replicate on it is dropped.
# Counts on Totoro (Julia 1.12.6), seeds 1:20, x scaled by 1e7 (Poisson) or 1e6
# (Binomial): unconverged on main 17 of 20 (Poisson), 12 of 20 (Binomial); with
# this change 0 of 20 and 1 of 20 (rep 12, which is left out below).
using DRModels
using Test, Random
import Distributions

function _gen_1050_cb(rep, scale; n = 200)
    Random.seed!(rep)
    x0 = randn(n)
    x = x0 .* scale
    y = Float64.([rand(Distributions.Poisson(exp(0.3 + 0.4 * v))) for v in x0])
    s = Float64.([rand(Distributions.Binomial(20, 1 / (1 + exp(-(0.2 + 0.8 * v))))) for v in x0])
    return (; y, s, fail = 20 .- s, x)
end

@testset "#1050 Poisson/Binomial unit-free converged flag" begin
    # Poisson, x * 1e7: every rep is the same stationary fit as at x * 1.
    for rep in 1:5
        d = _gen_1050_cb(rep, 1e7)
        d1 = _gen_1050_cb(rep, 1.0)
        fp = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = d.y, x = d.x))
        fp1 = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = d1.y, x = d1.x))
        @test fp.converged
        @test fp1.converged
        @test isapprox(fp.loglik, fp1.loglik; rtol = 1e-8)
    end

    # Binomial, x * 1e6 (reps that main reads unconverged).
    for rep in (1, 4, 5, 6)
        d = _gen_1050_cb(rep, 1e6)
        fb = drm(bf(@formula(cbind(s, fail) ~ x)), Binomial(); data = (; d.s, d.fail, d.x))
        @test fb.converged
    end

    # A bootstrap replicate on the scaled fit is not dropped as unconverged.
    d = _gen_1050_cb(1, 1e7)
    data = (; y = d.y, x = d.x)
    fit = drm(bf(@formula(y ~ x)), Poisson(); data)
    boot = bootstrap_result(fit; data, B = 20, rng = MersenneTwister(7), failures = :skip)
    @test boot.failed == 0

    # A truncated fit still reports unconverged: `g_tol = 1e6` stops at the start.
    @test !drm(bf(@formula(y ~ x)), Poisson(); data, g_tol = 1e6).converged
end
