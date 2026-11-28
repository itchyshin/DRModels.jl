# Cover `_conditional_dist`, `_nb2_conditional`, `_pit_obs`, and
# `_gamma_sigma_is_shape` in src/quantile_residuals.jl with closed-form checks.
using DRModels
using Test, Random
import Distributions

@testset "quantile_residuals: conditional distributions (closed forms)" begin
    μ_count = [2.5, 4.0]
    μ_prop = [0.3, 0.4]
    scales = Dict(:sigma => [0.8, 1.2], :trials => [10.0, 12.0], :nu => [5.0, 7.0])
    obs_prop = Dict(:mu => μ_prop)

  @testset "_nb2_conditional: NB2 vs Poisson limit (φ > 1e10·μ)" begin
        μ1, φ_small = 150.0, 1e9
        φ_large = 1.2e13
        d_small = DRModels._nb2_conditional(φ_small, μ1)
        d_large = DRModels._nb2_conditional(φ_large, μ1)
        @test d_small isa Distributions.NegativeBinomial
        @test d_large isa Distributions.Poisson
        @test d_large.λ ≈ μ1
        p = φ_small / (φ_small + μ1)
        @test Distributions.cdf(d_small, 10) ≈ Distributions.cdf(Distributions.NegativeBinomial(φ_small, p), 10)
    end

    @testset "_conditional_dist on discrete / proportion families" begin
        d_pois = DRModels._conditional_dist(Poisson(), 2; μ = μ_count, scales, obs = Dict(:mu => μ_count))
        @test d_pois isa Distributions.Poisson && d_pois.λ ≈ μ_count[2]

        d_tp = DRModels._conditional_dist(TruncatedPoisson(), 1; μ = μ_count, scales, obs = Dict(:mu => μ_count))
        @test d_tp isa Distributions.Poisson

        d_bin = DRModels._conditional_dist(Binomial(), 2; μ = μ_prop, scales, obs = obs_prop)
        @test d_bin isa Distributions.Binomial
        @test d_bin.n == 12
        @test d_bin.p ≈ μ_prop[2] atol = 1e-12

        d_bb = DRModels._conditional_dist(BetaBinomial(), 1; μ = μ_prop, scales, obs = obs_prop)
        φ = 1 / scales[:sigma][1]^2
        m = μ_prop[1]
        @test d_bb.n == 10
        @test Distributions.mean(d_bb) ≈ Distributions.mean(Distributions.BetaBinomial(10, m * φ, (1 - m) * φ))

        d_tnb = DRModels._conditional_dist(TruncatedNegBinomial2(), 1; μ = μ_count, scales, obs = Dict(:mu => μ_count))
        @test d_tnb isa Distributions.NegativeBinomial || d_tnb isa Distributions.Poisson
    end

    @testset "_conditional_dist on continuous families" begin
        d_st = DRModels._conditional_dist(Student(), 2; μ = μ_count, scales, obs = Dict(:mu => μ_count))
        @test d_st isa Distributions.LocationScale
        @test Distributions.location(d_st) ≈ μ_count[2]
        @test Distributions.scale(d_st) ≈ scales[:sigma][2]

        d_g_plain = DRModels._conditional_dist(Gamma(), 1; μ = μ_count, scales,
            obs = Dict(:mu => μ_count), gamma_sigma_is_shape = false)
        α = 1 / scales[:sigma][1]^2
        @test d_g_plain isa Distributions.Gamma
        @test d_g_plain.α ≈ α && d_g_plain.θ ≈ μ_count[1] / α

        d_g_shape = DRModels._conditional_dist(Gamma(), 1; μ = μ_count, scales,
            obs = Dict(:mu => μ_count), gamma_sigma_is_shape = true)
        @test d_g_shape.α ≈ scales[:sigma][1]
    end

    @testset "_pit_obs uses count × trials for Binomial / BetaBinomial" begin
        @test DRModels._pit_obs(Binomial(), 2; obs = obs_prop, scales) ==
              round(Int, μ_prop[2] * scales[:trials][2])
        @test DRModels._pit_obs(BetaBinomial(), 1; obs = obs_prop, scales) ==
              round(Int, μ_prop[1] * scales[:trials][1])
        @test DRModels._pit_obs(Poisson(), 1; obs = Dict(:mu => [3.0]), scales) == 3.0
    end

    @testset "_gamma_sigma_is_shape marks coupled LocScaleObjective Gamma only" begin
        G = 4; n = 12
        pair = repeat([40, 10, 30, 20], inner = 3)
        g = repeat(1:G, inner = n ÷ G)
        x = randn(n); y = abs.(1.0 .+ 0.2x .+ 0.1 .* randn(n))
        plain = drm(bf(@formula(y ~ x)), Gamma(); data = (; y, x))
        coupled = drm(bf(@formula(y ~ x + (1 | pair | g)), @formula(sigma ~ x + (1 | pair | g))),
                       Gamma(); data = (; y, x, g, pair))
        @test !DRModels._gamma_sigma_is_shape(plain)
        @test coupled.nll isa DRModels.LocScaleObjective
        @test DRModels._gamma_sigma_is_shape(coupled)
    end
end