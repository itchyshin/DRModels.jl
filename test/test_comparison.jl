# Model comparison + accessor parity: fit a Gaussian location–scale model where
# x is genuinely predictive (in both μ and log σ), then check the nested
# likelihood-ratio test, the AICc correction, prior weights, and `update`.
using DRModels
using Test, Random

@testset "comparison: lrtest / anova / aicc / weights / update" begin
    Random.seed!(20260604)
    n = 600
    x = randn(n)
    # x drives both the mean and the (log) scale, so dropping it should hurt.
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    data = (; y, x)

    full    = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)
    reduced = drm(bf(@formula(y ~ 1),     @formula(sigma ~ 1)),     Gaussian(); data = data)

    @testset "lrtest / anova" begin
        t = lrtest(reduced, full)
        @test t.statistic > 0
        @test t.dof == dof(full) - dof(reduced)
        @test 0 <= t.pvalue <= 1
        @test t.pvalue < 0.05               # x is truly predictive
        @test anova(reduced, full) == t     # alias

        # Wrong argument order (no extra parameters in "full") errors.
        @test_throws ArgumentError lrtest(full, reduced)
    end

    @testset "aicc" begin
        @test aicc(full) > aic(full)        # correction is strictly positive
        @test isfinite(aicc(full))
    end

    @testset "weights" begin
        @test weights(full) == ones(nobs(full))
        @test weights(reduced) == ones(nobs(reduced))
    end

    @testset "update" begin
        refit = update(full, bf(@formula(y ~ 1), @formula(sigma ~ 1)); data = data)
        @test refit isa DrmFit
        @test length(coef(refit)) == length(coef(reduced))
    end
end
