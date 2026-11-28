# DRModels.jl #1246 follow-ups: items 4, 5, and 6.
# `_va_scored_sigma` honours `:drm_quiet_boundary`; `_logfactorial` is O(1);
# `marginal = :VA` fits accept `confint(...; method = :profile)` on fixed effects.
using DRModels
using Test, Logging, Random
import Distributions

const DVI = DRModels

@testset "#1246 item 6: _logfactorial is O(1) and agrees with the sum" begin
    for k in 0:200
        @test DVI._logfactorial(k) ≈ sum(log, 2:k; init = 0.0) rtol = 1e-12
    end
    @test DVI._logfactorial(0) == 0.0
    @test DVI._logfactorial(1) == 0.0
    t = @elapsed v = DVI._logfactorial(10^9)
    @test isfinite(v) && v > 0
    @test t < 1.0
end

@testset "#1246 item 4: _va_scored_sigma clamp warning is quietable" begin
    b = 20.0
    Xσ = ones(5, 1)
    βσ = [b + 5.0]                       # η_σ past the clamp
    @test_logs (:warn,) DVI._va_scored_sigma(Xσ, βσ, b, "_fit_nb2_ranef_va")
    loud = with_logger(NullLogger()) do
        DVI._va_scored_sigma(Xσ, βσ, b, "_fit_nb2_ranef_va")
    end
    quiet = task_local_storage(:drm_quiet_boundary, true) do
        @test_logs DVI._va_scored_sigma(Xσ, βσ, b, "_fit_nb2_ranef_va")
    end
    @test quiet == loud == fill(exp(b), 5)
    # inside the band: silent either way, same value
    @test_logs DVI._va_scored_sigma(Xσ, [1.0], b, "_fit_nb2_ranef_va") == fill(exp(1.0), 5)
end

@testset "#1246 item 5: profile CIs on a Poisson marginal = :VA fit" begin
    rng = MersenneTwister(202611241)
    G = 12; per = 6; n = G * per
    g = repeat(1:G, inner = per); x = randn(rng, n)
    β = [0.35, -0.2]; σb = 0.45
    bg = σb .* randn(rng, G)
    λ = exp.(β[1] .+ β[2] .* x .+ bg[g])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + (1 | g))), DVI.Poisson();
              data = data, marginal = :VA, se = false)
    @test fit.marginal === :VA
    @test is_converged(fit)

    wald = confint(fit; method = :wald)
    prof = confint(fit; method = :profile)
    @test length(prof) == length(wald) == 3
    @test [r.param for r in prof] == [r.param for r in wald]
    @test all(r.lower < r.estimate < r.upper for r in prof)
    pres = profile_result(fit)
    @test pres.failed == 0
    @test pres.ci == prof
end
