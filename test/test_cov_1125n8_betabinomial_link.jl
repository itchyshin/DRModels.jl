# cov lane 1125n8 — BetaBinomial kernels and link_residual edges (src/betabinomial.jl, src/link_residual.jl).
using DRModels
using Test
using Distributions: BetaBinomial, logpdf
using SpecialFunctions: trigamma

@testset "cov 1125n8: _betabinomial_logit_ll" begin
    ref = logpdf(BetaBinomial(12, 6.0, 6.0), 4)
    η = 0.0
    φ = 12.0
    @test DRModels._betabinomial_logit_ll(12, 4, η, φ) ≈ ref rtol = 1e-10
    @test isfinite(DRModels._betabinomial_logit_ll(10, 0, -5.0, 3.0))
    @test isfinite(DRModels._betabinomial_logit_ll(10, 10, 5000.0, 4.0))
    ll1 = DRModels._betabinomial_logit_ll(20, 5, 0.2, 8.0)
    ll2 = DRModels._betabinomial_logit_ll(20, 5, 0.2 + 1e-8, 8.0)
    @test ll1 ≈ ll2 rtol = 1e-6
end

@testset "cov 1125n8: cbind marker" begin
    x = cbind([1, 2], [3, 4])
    @test size(x) == (2, 2)
    @test x[1, 1] == 1 && x[1, 2] == 3
end

@testset "cov 1125n8: link_residual guards" begin
    @test DRModels.link_residual(DRModels.Poisson(), 1e-20) ≈ log1p(1 / 1e-12)
    @test DRModels.link_residual(DRModels.Gaussian(); dispersion = 2.5) == 2.5
    @test DRModels.link_residual(DRModels.Beta(), 0.25; dispersion = 10.0) ≈
          trigamma(2.5) + trigamma(7.5)
    @test_throws UndefKeywordError DRModels.link_residual(DRModels.Beta(), 0.3)
end

@testset "cov 1125n8: coevolution accessors on mock fit" begin
    Σ = [1.0 0.2 0.1 0.0
         0.2 1.0 0.0 0.1
         0.1 0.0 0.5 0.05
         0.0 0.1 0.05 0.5]
    re = (Sigma_a = Σ, axes = (:mu1, :mu2, :sigma1, :sigma2))
    fit = DrmFit(
        Gaussian(),
        [:mu1 => 1:1],
        [:mu1 => ["(Intercept)"]],
        [0.0], [1.0;;], 0.0, 4, true,
        Dict(:mu1 => zeros(4)),
        Dict(:mu1 => ones(4)),
        Dict(:sigma1 => ones(4)),
        nothing, nothing, nothing, re, :ML, NaN, 0.0, :LA,
    )
    vc = DRModels.coevolution_vc(fit)
    @test vc.sd[:mu1] ≈ 1.0
    cor = DRModels.coevolution_cor(fit)
    @test cor.cor[1, 1] == 1.0
    @test cor.cor[1, 2] ≈ 0.2 rtol = 1e-12
    sm = DRModels.coevolution_summary(fit)
    @test length(sm.pair) == 6
    bad = DrmFit(Poisson(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], [1.0;;],
                 0.0, 1, true, Dict(:mu => [0.0]), Dict(:mu => [0.0]), Dict(:sigma => [1.0]))
    @test_throws ErrorException DRModels.coevolution_cor(bad)
end