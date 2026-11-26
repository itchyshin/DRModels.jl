# Issue #1534: `using DRModels, Distributions` must allow unqualified family/distribution
# names for Poisson, Gamma, Beta, LogNormal, and Binomial() → drm (default logit).

using DRModels, Distributions, Random, Test

@testset "DRModels + Distributions family constructors (#1534)" begin
  Random.seed!(26)
  n = 80
  x = randn(n)
  y = Float64[rand(Distributions.Poisson(exp(0.2 + 0.5 * xi))) for xi in x]

  @testset "unqualified Poisson/Gamma/Beta/LogNormal resolve and simulate" begin
    @test rand(Poisson(2.0)) isa Integer
    @test Poisson() isa Poisson
    @test Gamma() isa Gamma
    @test Beta() isa Beta
    @test LogNormal() isa LogNormal
  end

  @testset "drm fixed-effect Poisson with shared name" begin
    fit = drm(bf(@formula(y ~ x)), Poisson(); data = (; y, x))
    @test fit.family isa Poisson
    @test is_converged(fit)
    @test coef(fit, :mu) isa AbstractVector
  end

  @testset "Binomial: Distributions.Binomial() for drm (logit)" begin
    yb = rand(Distributions.Binomial(1, 0.4), n)
    fit = drm(bf(@formula(yb ~ x)), Distributions.Binomial(); data = (; yb, x))
    @test fit.family isa DRModels.Binomial
    @test fit.family.link === :logit
  end

  @testset "Binomial probit via DRModels.Binomial when Distributions is loaded" begin
    yb = rand(Distributions.Binomial(1, 0.4), n)
    fit = drm(bf(@formula(yb ~ x)), DRModels.Binomial(link = :probit); data = (; yb, x))
    @test fit.family.link === :probit
  end
end