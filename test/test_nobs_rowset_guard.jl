# #1053: lrtest / anova / lrt_boundary must refuse two fits that share nobs
# but were estimated on different rows. The count-only guard let that pair
# through (review of #1044). Write this as a failing test first; the guard
# then compares the observation row set, not only the integer count.
using DRModels
using Test
using Random

function _argmsg(f)
    try
        f()
        error("expected an ArgumentError")
    catch err
        err isa ArgumentError || rethrow()
        return err.msg
    end
end

@testset "#1053 nobs guard compares row sets, not only counts" begin
    Random.seed!(1053)
    n = 80
    x = randn(n)
    y = 0.4 .+ 0.7 .* x .+ 0.4 .* randn(n)
    # Distinct values at the two drop sites so a value-only fingerprint cannot
    # collide when those rows are the ones removed.
    y[3] = 10.0
    y[10] = -10.0
    g = repeat(1:16, inner = 5)

    @testset "same nobs, missing response on different rows" begin
        ya = copy(y); ya[3] = NaN
        yb = copy(y); yb[10] = NaN
        data_a = (; y = ya, x)
        data_b = (; y = yb, x)
        full = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = data_a)
        reduced = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data = data_b)
        @test nobs(full) == nobs(reduced) == n - 1
        msg = _argmsg(() -> lrtest(reduced, full))
        @test occursin("different samples", msg)
        @test occursin("nobs", msg)
        @test occursin("lrtest", msg)
        amsg = _argmsg(() -> anova(reduced, full))
        @test occursin("different samples", amsg)
        @test occursin("anova", amsg) || occursin("lrtest", amsg)
    end

    @testset "same nobs, missing predictor on different rows" begin
        xa = Vector{Union{Missing,Float64}}(x); xa[3] = missing
        xb = Vector{Union{Missing,Float64}}(x); xb[10] = missing
        fit_a = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian();
                    data = (; y, x = xa))
        fit_b = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian();
                    data = (; y, x = xb))
        @test nobs(fit_a) == nobs(fit_b) == n - 1
        # Same formula (equal dof) so the public lrtest would refuse for the
        # parameter-count reason; the guard itself must still see different rows.
        msg = _argmsg(() -> DRModels._nobs_compare_guard(fit_a, fit_b, "lrtest"))
        @test occursin("different samples", msg)
        @test occursin("nobs", msg)
    end

    @testset "same rows still compare" begin
        ya = copy(y); ya[3] = NaN
        data = (; y = ya, x)
        full = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data)
        reduced = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data)
        @test nobs(full) == nobs(reduced) == n - 1
        t = lrtest(reduced, full)
        @test t.dof == 1
        @test anova(reduced, full) == t
    end

    @testset "lrt_boundary sibling uses the same row-set guard" begin
        ya = copy(y); ya[3] = NaN
        yb = copy(y); yb[10] = NaN
        data_a = (; y = ya, x, g)
        data_b = (; y = yb, x, g)
        full = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                   data = data_a)
        reduced = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                      data = data_b)
        @test nobs(full) == nobs(reduced)
        msg = _argmsg(() -> lrt_boundary(full, reduced; q = 1))
        @test occursin("lrt_boundary", msg)
        @test occursin("different samples", msg)
        @test occursin("nobs", msg)
    end

    @testset "lrt_boundary same missing row still compares (RE vs FE storage)" begin
        # Gaussian (1 | g) listwise-drops a missing y; the fixed-effect route
        # keeps a NaN on the original length. Same rows must still compare.
        ya = copy(y); ya[3] = NaN
        data = (; y = ya, x, g)
        full = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                   data)
        reduced = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                      data)
        @test nobs(full) == nobs(reduced) == n - 1
        t = lrt_boundary(full, reduced; q = 1)
        @test t.q == 1
        @test isfinite(t.statistic)
    end
end
