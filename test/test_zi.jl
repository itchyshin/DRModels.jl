# Zero-inflation modifier `zi` for the count families. A `zi ~ …` formula adds a
# structural-zero component (logit link): P(0) = π + (1-π)·P_count(0),
# P(k>0) = (1-π)·P_count(k). Applies to Poisson (ZIP) and NB2 (ZINB). Mirrors
# drmTMB's `zi` modifier.
using DRModels
using Test, Random
import Distributions

@testset "Zero-inflation modifier (zi)" begin
    @testset "ZIP — Poisson + zi recovery" begin
        Random.seed!(20260619)
        n = 4000; x = randn(n)
        β = [0.6, 0.4]; ηπ = -0.4                  # logit π = -0.4 ⇒ π ≈ 0.40
        π = 1 / (1 + exp(-ηπ))
        λ = exp.(β[1] .+ β[2] .* x)
        y = Float64.([rand() < π ? 0 : rand(Distributions.Poisson(λi)) for λi in λ])
        fit = drm(bf(@formula(y ~ x), @formula(zi ~ 1)), Poisson(); data = (; y, x))
        @test coef(fit, :mu)[1] ≈ β[1] atol = 0.10
        @test coef(fit, :mu)[2] ≈ β[2] atol = 0.08
        @test 1 / (1 + exp(-coef(fit, :zi)[1])) ≈ π atol = 0.10   # zero-inflation prob
        @test isfinite(loglik(fit))
    end

    @testset "ZINB — negative-binomial + zi recovery" begin
        Random.seed!(20260620)
        n = 4000; x = randn(n)
        β = [0.6, 0.4]; θ = 3.0; πz = 0.30
        μ = exp.(β[1] .+ β[2] .* x)
        y = Float64.([rand() < πz ? 0 : rand(Distributions.NegativeBinomial(θ, θ / (θ + μi))) for μi in μ])
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zi ~ 1)), NegBinomial2(); data = (; y, x))
        @test coef(fit, :mu)[1] ≈ β[1] atol = 0.12
        @test coef(fit, :mu)[2] ≈ β[2] atol = 0.10
        @test 1 / (1 + exp(-coef(fit, :zi)[1])) ≈ πz atol = 0.12
        @test isfinite(loglik(fit))
    end
end
