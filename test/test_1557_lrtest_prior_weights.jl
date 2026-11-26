# Refs #1557: lrtest / anova must refuse when prior `weights` differ (drmTMB #1888).
using DRModels, StatsModels, Test

@testset "lrtest / anova refuse mismatched prior weights (#1557)" begin
    data = (
        y = [-0.42, 0.18, 1.05, -0.77, 0.33, -1.12, 0.91, 0.05, -0.28, 0.64,
             -0.55, 1.22, 0.09, -0.86, 0.47, -0.19, 0.73, -1.01, 0.36, 0.58,
             -0.31, 0.82, -0.64, 0.11, 0.29, -0.48, 0.67, -0.22, 0.95, -0.71,
             0.14, -0.39, 0.52, -0.93, 0.27, 0.61, -0.16, 0.44, -0.57, 0.79],
        x = [-0.9, 0.3, 1.1, -0.5, 0.7, -1.2, 0.8, 0.0, -0.4, 0.6,
             -0.7, 1.3, 0.1, -0.8, 0.5, -0.2, 0.9, -1.0, 0.4, 0.55,
             -0.35, 0.85, -0.6, 0.15, 0.25, -0.45, 0.65, -0.25, 1.0, -0.75,
             0.12, -0.38, 0.5, -0.95, 0.2, 0.6, -0.18, 0.42, -0.55, 0.78],
    )
    w = fill(1.0, 40)
    w[2:2:40] .= 2.0

    reduced = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data = data, weights = w)
    full = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = data)

    @test_throws ArgumentError lrtest(reduced, full)
    @test_throws ArgumentError anova(reduced, full)
    err = try
        lrtest(reduced, full)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("prior weights", err.msg)

    full_w = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = data, weights = w)
    t = lrtest(reduced, full_w)
    @test t.statistic > 0
    @test 0 <= t.pvalue <= 1

    unweighted_red = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data = data)
    unweighted_full = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = data)
    ones_full = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian();
                    data = data, weights = ones(40))
    t_unw = lrtest(unweighted_red, unweighted_full)
    @test isfinite(t_unw.statistic)
    t_ones = lrtest(unweighted_red, ones_full)
    @test isfinite(t_ones.statistic)

    w2 = copy(w)
    w2[1] = 3.0
    mismatched = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian();
                    data = data, weights = w2)
    @test_throws ArgumentError lrtest(reduced, mismatched)
end