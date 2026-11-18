# test_lrtest_statsmodels_shared.jl -- DRModels and StatsModels share one `lrtest`.
#
# DRModels used to define its own `lrtest` and export it. StatsModels exports a function
# of the same name, so `using DRModels, StatsModels; lrtest(...)` was an ambiguous
# binding (UndefVarError on Julia 1.12 and later). DRModels now adds a method to
# `StatsModels.lrtest` for `DrmFit` only. This file loads ALL of StatsModels on purpose.
using DRModels, StatsModels, Test

@testset "lrtest is one function shared with StatsModels" begin
    # Fixed, RNG-free data. The reference numbers below were computed on main
    # (e7d5ddee1, before the change) with `DRModels.lrtest(reduced, full)`.
    x = collect(range(-1.5, 1.5; length = 60))
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* sin.(7 .* (1:60) .^ 2 ./ 13.0)
    data = (; y, x)
    full    = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data)
    reduced = drm(bf(@formula(y ~ 1),     @formula(sigma ~ 1)),     Gaussian(); data)

    @testset "one function" begin
        @test DRModels.lrtest === StatsModels.lrtest
        @test DRModels.anova(reduced, full) == lrtest(reduced, full)
    end

    @testset "unqualified call with both packages loaded" begin
        t = lrtest(reduced, full)
        @test t isa NamedTuple{(:statistic, :dof, :pvalue), Tuple{Float64, Int, Float64}}
        @test t.dof == 2
        @test t.statistic ≈ 66.86086632463122 rtol = 1e-8
        @test t.pvalue ≈ 3.0293353305261703e-15 rtol = 1e-6
        @test startswith(sprint(show, t), "(statistic = ")
    end

    @testset "DRModels refusals still come from DRModels" begin
        # reduced/full swapped: DRModels' own argument error, not a StatsModels one.
        @test_throws ArgumentError lrtest(full, reduced)
    end

    @testset "StatsModels method set is untouched for other models" begin
        ms = methods(StatsModels.lrtest)
        drm_ms = filter(m -> m.sig <: Tuple{Any, DrmFit, DrmFit}, collect(ms))
        @test length(drm_ms) == 1
        @test !hasmethod(lrtest, Tuple{DrmFit})  # no DrmFit-only method; StatsModels' needs StatisticalModel
        @test !(DrmFit <: StatsModels.StatisticalModel)
    end
end
