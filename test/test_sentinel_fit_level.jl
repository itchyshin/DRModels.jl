# A fit stranded on the 1e18 failed-objective sentinel has loglik = -1e18: finite,
# and (zero-gradient plateau) possibly `converged`. Fit-level consumers must refuse
# it; healthy fits must be unchanged.
using DRModels
using Test, Random, Logging

# Rebuild a DrmFit with `loglik` (and `ml_loglik`) replaced; every other field kept.
function _with_loglik(f, ll)
    vals = Any[getfield(f, k) for k in fieldnames(typeof(f))]
    vals[findfirst(==(:loglik), fieldnames(typeof(f)))] = ll
    vals[findfirst(==(:ml_loglik), fieldnames(typeof(f)))] = ll
    vals[findfirst(==(:converged), fieldnames(typeof(f)))] = true   # plateau "converged"
    return typeof(f).name.wrapper(vals...)
end

@testset "sentinel fit-level guards" begin
    Random.seed!(20260929)
    n = 300
    x = randn(n)
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    data = (; y, x)
    full    = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data)
    reduced = drm(bf(@formula(y ~ 1),     @formula(sigma ~ 1)),     Gaussian(); data)
    bad = _with_loglik(full, -1e18)
    # The constructor clears `converged` when loglik is the -1e18 sentinel
    # (#1009 / #1012 / #1019). `_with_loglik` still asks for `true`.
    @test !bad.converged
    @test DRModels._sentinel_loglik(bad)
    @test !DRModels._nondegenerate_fit(bad)
    @test !is_converged(bad)

    @testset "healthy fits unchanged" begin
        @test !DRModels._sentinel_loglik(full)
        @test DRModels._nondegenerate_fit(full)
        t = lrtest(reduced, full)
        @test t.statistic ≈ 2 * (loglik(full) - loglik(reduced)) atol = 1e-12
        @test aic(full) ≈ -2 * loglik(full) + 2 * length(full.theta) atol = 1e-12
        @test bic(full) ≈ -2 * loglik(full) + length(full.theta) * log(nobs(full)) atol = 1e-12
        k = dof(full)
        @test aicc(full) ≈ aic(full) + 2k * (k + 1) / (nobs(full) - k - 1) atol = 1e-12
    end

    @testset "sentinel fit refused" begin
        @test_throws ArgumentError lrtest(reduced, bad)
        @test_throws ArgumentError anova(reduced, bad)
        @test_throws ArgumentError lrt_boundary(bad, reduced)
        @test_throws ArgumentError lrt_boundary(full, _with_loglik(reduced, -1e18))
        with_logger(NullLogger()) do
            @test isnan(aic(bad))
            @test isnan(bic(bad))
            @test isnan(aicc(bad))
        end
        @test_logs (:warn, r"degenerate") aic(bad)
    end
end
