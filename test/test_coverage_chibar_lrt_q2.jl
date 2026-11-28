# Live `lrt_boundary(; q = 2)` and extra χ̄² tails in src/chibar.jl.
using DRModels
using Test, Random, Statistics
import Distributions: Chisq, ccdf

@testset "chibar / lrt_boundary coverage (q = 2)" begin
    @testset "chibar_pvalue q = 2: negative statistic clamp" begin
        @test chibar_pvalue(-1.0, 2) == 0.75
        @test chibar_pvalue(-1.0, 1) == 0.5
    end

    @testset "lrt_boundary with two nested random intercepts (q = 2)" begin
        Random.seed!(202611201)
        G = 40; m = 10; n = G * m
        g = repeat(1:G, inner = m)
        h = repeat(1:4, inner = n ÷ 4)
        x = randn(n)
        y = 0.2 .- 0.3 .* x .+ 0.4 .* randn(n) .+
            0.0 .* (randn(G)[g] .+ randn(4)[h])  # true null for both RE
        data = (; y, x, g, h)

        full = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                    Gaussian(); data = data)
        reduced = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = data)

        t = lrt_boundary(full, reduced; q = 2)
        @test t.q == 2
        @test t.statistic ≈ 2 * (loglik(full) - loglik(reduced))
        @test t.pvalue == chibar_pvalue(t.statistic, 2)
        @test t.pvalue_naive == ccdf(Chisq(2), max(t.statistic, 0.0))
        @test t.pvalue <= t.pvalue_naive
        if t.statistic > 0
            naive1 = ccdf(Chisq(1), t.statistic)
            @test t.pvalue >= chibar_pvalue(t.statistic, 1)
            @test t.pvalue <= naive1
        end
    end
end