# Chi-bar-square boundary-corrected p-values for variance-component LR tests.
#
# Testing a variance component = 0 is a BOUNDARY problem: the LR statistic
# 2·(ℓ_full − ℓ_reduced) follows a chi-bar-square mixture, not χ²(df). For one
# boundary parameter the null is 0.5·χ²(0) + 0.5·χ²(1), so p = 0.5·P(χ²₁ > stat).
# These tests check (1) the q=1 closed form, (2) the q=2 mixture, and (3) that on
# a true-null simulation the boundary p-values are better-calibrated than the
# naive χ²(1) p-values (which are conservative on the boundary).
using DRModels
using Test, Random, Statistics
using Distributions: Chisq, ccdf

@testset "chi-bar-square boundary p-values" begin

    @testset "chibar_pvalue q=1 closed form" begin
        # For stat > 0 the χ̄² p-value is exactly half the naive χ²(1) tail.
        for stat in (0.25, 1.0, 2.71, 3.84, 6.63, 10.0)
            @test chibar_pvalue(stat, 1) == 0.5 * ccdf(Chisq(1), stat)
            @test chibar_pvalue(stat, 1) ≈ 0.5 * ccdf(Chisq(1), stat)
        end
        # At the boundary (stat ≈ 0) the continuous part gives ~0.5 (the χ²(0)
        # atom is a point mass and does not add to the upper tail).
        @test chibar_pvalue(0.0, 1) == 0.5
        @test chibar_pvalue(1e-10, 1) ≈ 0.5 atol = 1e-4
        # Default q is 1.
        @test chibar_pvalue(3.84) == chibar_pvalue(3.84, 1)
        # The correction is anti-conservative relative to naive χ²(1): smaller p.
        @test chibar_pvalue(3.84, 1) < ccdf(Chisq(1), 3.84)
        # Negative statistic clamps to the boundary value.
        @test chibar_pvalue(-2.0, 1) == 0.5
        # A valid p-value in [0, 1], monotone decreasing in the statistic.
        @test 0.0 <= chibar_pvalue(5.0, 1) <= 1.0
        @test chibar_pvalue(5.0, 1) < chibar_pvalue(1.0, 1)
    end

    @testset "chibar_pvalue q=2 mixture (0.25/0.5/0.25)" begin
        # Two independent boundary params: 0.25·χ²(0) + 0.5·χ²(1) + 0.25·χ²(2).
        for stat in (0.5, 2.0, 4.0, 5.99, 9.21)
            expected = 0.5 * ccdf(Chisq(1), stat) + 0.25 * ccdf(Chisq(2), stat)
            @test chibar_pvalue(stat, 2) == expected
        end
        # Boundary value: 0.5 + 0.25 = 0.75 at stat = 0.
        @test chibar_pvalue(0.0, 2) == 0.75
        # Strictly between the naive χ²(2) tail and the q=1 χ̄² p-value, and below
        # the naive χ²(2) reference (correction is anti-conservative).
        @test chibar_pvalue(5.99, 2) < ccdf(Chisq(2), 5.99)
        # Unsupported q errors.
        @test_throws ArgumentError chibar_pvalue(3.0, 3)
        @test_throws ArgumentError chibar_pvalue(3.0, 0)
    end

    @testset "lrt_boundary on a fitted random-intercept model" begin
        # x is in the mean; g carries a genuine random intercept. Dropping (1|g)
        # removes ONE variance component → q = 1 boundary test.
        Random.seed!(20260610)
        G = 60; m = 20; n = G * m
        g = repeat(1:G, inner = m); x = randn(n)
        σb = 0.8
        b = σb .* randn(G)
        y = 0.5 .- 0.4 .* x .+ b[g] .+ 0.7 .* randn(n)
        data = (; y, x, g)

        full    = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = data)
        reduced = drm(bf(@formula(y ~ x),           @formula(sigma ~ 1)), Gaussian(); data = data)

        t = lrt_boundary(full, reduced; q = 1)
        @test t.statistic ≈ 2 * (loglik(full) - loglik(reduced))
        @test t.q == 1
        @test t.pvalue == chibar_pvalue(t.statistic, 1)
        @test t.pvalue_naive == ccdf(Chisq(1), max(t.statistic, 0.0))
        # The boundary p-value is exactly half the naive one (q = 1, stat > 0):
        @test t.statistic > 0
        @test t.pvalue ≈ 0.5 * t.pvalue_naive
        @test t.pvalue <= t.pvalue_naive
        # Real random effect → both reject; boundary p-value at least as small.
        @test t.pvalue < 0.05
    end

    @testset "null calibration: χ̄² beats naive χ²(1)" begin
        # TRUE NULL: no group effect (σ_b = 0). Under the boundary, the LR
        # statistic is ~0.5 point-mass + 0.5·χ²(1). The naive χ²(1) p-value treats
        # all of it as continuous χ²(1), so it is stochastically too LARGE
        # (conservative) → it rejects FAR below the nominal α. The χ̄² p-value
        # halves the continuous tail and is calibrated → rejects near α.
        α = 0.05
        nrep = 120
        pcorr = Vector{Float64}(undef, nrep)
        pnaive = Vector{Float64}(undef, nrep)
        for i in 1:nrep
            Random.seed!(7000 + i)
            G = 25; m = 12; nn = G * m
            gg = repeat(1:G, inner = m); xx = randn(nn)
            # NO b[gg] term: the random-intercept variance is exactly zero.
            yy = 0.4 .- 0.5 .* xx .+ 0.7 .* randn(nn)
            d = (; y = yy, x = xx, g = gg)
            full    = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d)
            reduced = drm(bf(@formula(y ~ x),           @formula(sigma ~ 1)), Gaussian(); data = d)
            t = lrt_boundary(full, reduced; q = 1)
            pcorr[i] = t.pvalue
            pnaive[i] = t.pvalue_naive
        end
        rej_corr = mean(pcorr .< α)
        rej_naive = mean(pnaive .< α)

        # By construction p_corr = 0.5 * p_naive whenever stat > 0, so the
        # boundary test rejects at least as often as naive — never worse.
        @test rej_corr >= rej_naive
        # Naive χ²(1) is conservative under the boundary: its rejection rate sits
        # well below the nominal α (it should be roughly α/2 asymptotically).
        @test rej_naive <= α
        # The boundary rejection rate is closer to the nominal α than naive's
        # (better calibrated). Both are bounded above by α here (finite sample),
        # so "closer to α" means "larger but still ≤ α-ish band".
        @test abs(rej_corr - α) <= abs(rej_naive - α)
        # The point mass at 0: a large fraction of statistics are ~0, the
        # chi-bar-square signature. (Asymptotically ~0.5; small-sample a bit
        # higher.) p_naive = ccdf(χ²₁, stat) ≈ 1 exactly when stat ≈ 0, so a high
        # fraction of p_naive ≈ 1 witnesses the atom that makes naive χ²(1) wrong.
        frac_atom = mean(pnaive .> 0.99)
        @test frac_atom >= 0.30
    end
end
