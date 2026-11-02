# #926 remaining: the marginal bootstrap simulator treated `sigma ~ (1 | g)`
# as constant σ. `scales[:sigma]` is the population (b = 0) scale; the
# log-σ intercept must be redrawn so group residual SDs vary across draws.
using DRModels
using Test, Random, Statistics

function _926_group_residual_sds(s, mu, g, G)
    return [std(s[g .== k] .- mu[g .== k]) for k in 1:G]
end

@testset "scale random intercept is redrawn in the bootstrap simulator (#926)" begin
    Random.seed!(926)
    G = 24
    m = 16
    n = G * m
    g = repeat(1:G, inner = m)
    x = randn(n)

    @testset "sigma-only: y ~ x, sigma ~ 1 + (1 | g)" begin
        τ = 0.7
        bg = τ .* randn(G)
        y = 0.4 .+ 0.5 .* x .+ exp.(log(0.5) .+ bg[g]) .* randn(n)
        dat = (; y, x, g)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g))), Gaussian();
                  data = dat)
        @test fit.converged
        sim = DRModels._marginal_simulator(fit, dat)
        @test sim !== nothing
        mu = fit.means[:mu]
        rng = MersenneTwister(9261)
        spreads = [std(_926_group_residual_sds(sim(rng), mu, g, G)) for _ in 1:80]
        # Constant-σ draws have only sampling noise in the group SDs
        # (order σ/sqrt(2(m-1)) ≈ 0.09). A redrawn log-σ intercept of 0.7
        # produces a much larger spread of group residual SDs.
        @test median(spreads) > 0.15
    end

    @testset "mean + sigma: y ~ x + (1 | g), sigma ~ 1 + (1 | g)" begin
        bμ = 0.5 .* randn(G)
        bσ = 0.6 .* randn(G)
        y = 0.3 .+ 0.4 .* x .+ bμ[g] .+ exp.(log(0.5) .+ bσ[g]) .* randn(n)
        dat = (; y, x, g)
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g))),
                  Gaussian(); data = dat)
        @test fit.converged
        sim = DRModels._marginal_simulator(fit, dat)
        @test sim !== nothing
        mu = Vector{Float64}(predict(fit, dat))
        rng = MersenneTwister(9262)
        spreads = [std(_926_group_residual_sds(sim(rng), mu, g, G)) for _ in 1:80]
        @test median(spreads) > 0.12
    end
end
