# Round 3 coverage — src/chibar.jl (boundary LR p-values and lrt_boundary).
using DRModels, Test, Random
using DRModels: Poisson, NegBinomial2
using Distributions: Chisq, ccdf

@testset "cov 1125x3 chibar_pvalue" begin
    stats = (-1.0, 0.0, 0.01, 1.5, 4.0, 12.0)
    for s in stats
        sc = max(s, 0.0)
        @test chibar_pvalue(s, 1) ≈ 0.5 * ccdf(Chisq(1), sc) atol = 1e-14
        @test chibar_pvalue(s, 2) ≈ 0.5 * ccdf(Chisq(1), sc) + 0.25 * ccdf(Chisq(2), sc) atol = 1e-14
    end
    @test chibar_pvalue(0.0, 1) == 0.5
    @test chibar_pvalue(0.0, 2) == 0.75
    @test chibar_pvalue(-3.0, 1) == 0.5
    @test chibar_pvalue(-3.0, 2) == 0.75
    for q in (0, 3, -1, 5)
        @test_throws ArgumentError chibar_pvalue(1.0, q)
    end
    err = try chibar_pvalue(1.0, 3); nothing catch e; e end
    @test err isa ArgumentError && occursin("q = 1 or q = 2", err.msg)
    @test chibar_pvalue(1.0, 1) <= ccdf(Chisq(1), 1.0)
    @test chibar_pvalue(1.0, 2) <= ccdf(Chisq(2), 1.0)
    @test chibar_pvalue(8.0, 1) > 0
    @test chibar_pvalue(8.0, 1) < 0.01
end

@testset "cov 1125x3 lrt_boundary Gaussian RI" begin
    rng = MersenneTwister(112503)
    g = repeat(1:10, inner = 5)
    x = randn(rng, 50)
    y = 0.3 .+ 0.4 .* x .+ 0.9 .* randn(rng, 10)[g] .+ 0.2 .* randn(rng, 50)
    dat = (; y, x, g)
    full = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dat)
    reduced = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = dat)
    t1 = lrt_boundary(full, reduced; q = 1)
    @test t1.q == 1
    @test t1.statistic ≈ 2 * (loglik(full) - loglik(reduced)) atol = 1e-10
    @test t1.pvalue ≈ chibar_pvalue(t1.statistic, 1) atol = 1e-14
    @test t1.pvalue_naive ≈ ccdf(Chisq(1), max(t1.statistic, 0.0)) atol = 1e-14
    @test t1.pvalue <= t1.pvalue_naive
    @test isfinite(t1.pvalue) && isfinite(t1.statistic)
end

@testset "cov 1125x3 lrt_boundary count family nest" begin
    rng = MersenneTwister(112504)
    n = 200
    x = randn(rng, n)
    λ = exp.(0.2 .+ 0.25 .* x)
    y = [rand(rng) < min(λ[i], 0.999) ? 1 : 0 for i in 1:n]
    dat = (; y, x)
    nb = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = dat)
    po = drm(bf(@formula(y ~ x)), Poisson(); data = dat)
    t = lrt_boundary(nb, po; q = 1)
    @test t.q == 1
    @test isfinite(t.statistic)
    @test 0 < t.pvalue <= 1
    @test t.pvalue <= t.pvalue_naive
    @test t.pvalue ≈ chibar_pvalue(t.statistic, 1) atol = 1e-12
end