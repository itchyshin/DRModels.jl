# test_lrtest_boundary_warn.jl — issue #304.
#
# lrtest/anova use a naive χ²(Δdof) reference, which is INVALID when the dropped
# parameter is a variance component (the tested null variance = 0 sits on the
# boundary; the correct reference is a chi-bar-square mixture). lrtest must detect
# this case and warn, pointing the user to the boundary-corrected `lrt_boundary`.
# A non-variance (mean / scale coefficient) drop must NOT warn.
#
# Uses only Test macros (@test_logs / @test_warn) — no extra Logging dependency.
using DRModels, Test, Random

@testset "lrtest boundary variance-component warning (#304)" begin
    Random.seed!(20260703)
    G = 30; m = 12; nn = G * m
    g = repeat(1:G, inner = m); x = randn(nn)
    b = 0.8 .* randn(G)
    y = 0.5 .- 0.4 .* x .+ b[g] .+ 0.7 .* randn(nn)
    d = (; y, x, g)

    full_re    = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
    reduced_re = drm(bf(@formula(y ~ x),           @formula(sigma ~ 1)), Gaussian(); data = d)

    @testset "dropping a random effect warns" begin
        # @test_logs matches a :warn record; @test_warn checks the message text.
        @test_logs (:warn,) match_mode = :any lrtest(reduced_re, full_re)
        @test_warn "lrt_boundary" lrtest(reduced_re, full_re)
        # lrtest still returns the naive result (behaviour preserved; only warned).
        # Capture the :warn so it does not print, and inspect the returned tuple.
        t = (@test_logs (:warn,) match_mode = :any lrtest(reduced_re, full_re))
        @test t.dof == dof(full_re) - dof(reduced_re)
        @test 0 <= t.pvalue <= 1
        # Boundary-corrected p-value is available and smaller (less conservative).
        tb = lrt_boundary(full_re, reduced_re; q = 1)
        @test tb.pvalue <= t.pvalue
    end

    @testset "dropping only mean/scale coefficients does NOT warn" begin
        full_fx    = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = d)
        reduced_fx = drm(bf(@formula(y ~ 1), @formula(sigma ~ 1)), Gaussian(); data = d)
        # No log records at all for a purely interior-parameter comparison.
        t = (@test_logs lrtest(reduced_fx, full_fx))
        @test 0 <= t.pvalue <= 1
    end

    @testset "anova alias warns the same way" begin
        @test_logs (:warn,) match_mode = :any anova(reduced_re, full_re)
    end
end
