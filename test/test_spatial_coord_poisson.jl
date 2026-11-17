# test_spatial_coord_poisson.jl — coordinate-based exponential-kernel spatial covariance for
# POISSON counts. Since wave 10 the range is FIXED at drmTMB's value (the median positive
# pairwise distance; kernel exp(-d / r) + 1e-6 I), as for Gamma and the other count routes, so
# the fit has the parameters drmTMB has (beta, log SD). Agreement with drmTMB 0.8.1 is in
# test_spatial_parity_wave10.jl.
#
# Two checks:
#   1. Recovery: fixed effects and the spatial SD are recovered from data simulated from the
#      same fixed-range kernel the fit uses.
#   2. Route identity: `coords = …` gives the same fit as the precomputed `K = ` route given
#      the native kernel, to floating point.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

# Native fixed-range kernel among G sites at random positions.
function _spatial_cov(rng, G)
    pos = rand(rng, G, 2) .* 10.0
    return pos, DRModels._gaussian_spatial_sd_kernel(pos, G, :site)
end

@testset "Poisson spatial(1|site, coords): fixed-range kernel, recovery" begin
    rng = MersenneTwister(20260610)
    G = 90
    m = 6
    σb = 0.55
    coords, C = _spatial_cov(rng, G)
    site = repeat(1:G, inner = m)
    n = length(site)
    x = randn(rng, n)
    β = [0.20, 0.40]
    u = σb .* (cholesky(Symmetric(C)).L * randn(rng, G))
    λ = exp.(β[1] .+ β[2] .* x .+ u[site])
    y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
    data = (; y, x, site)

    fit = drm(bf(@formula(y ~ x + spatial(1 | site))), Poisson();
              data = data, coords = coords)

    @test fit.converged
    @test !(:range in first.(fit.blocks))
    @test coef(fit, :mu)[2] ≈ β[2] atol = 0.15          # slope (RE-independent)
    @test re_sd(fit)[:site] ≈ σb atol = 0.20            # spatial SD recovered
    @test isfinite(loglik(fit))
    @test all(fitted(fit; re = :zero) .> 0)
end

@testset "Poisson spatial coords: same fit as the K = native kernel route" begin
    rng = MersenneTwister(20260612)
    G = 25
    m = 5
    coords, C = _spatial_cov(rng, G)
    site = repeat(1:G, inner = m)
    n = length(site)
    x = randn(rng, n)
    u = 0.5 .* (cholesky(Symmetric(C)).L * randn(rng, G))
    y = Float64.([rand(rng, Distributions.Poisson(exp(0.2 + 0.35x[i] + u[site[i]]))) for i in 1:n])
    data = (; y, x, site)
    fit_c = drm(bf(@formula(y ~ x + spatial(1 | site))), Poisson(); data, coords, se = false)
    fit_K = drm(bf(@formula(y ~ x + spatial(1 | site))), Poisson(); data, K = Matrix(C), se = false)
    @test loglik(fit_c) ≈ loglik(fit_K) atol = 1e-6
    @test coef(fit_c, :mu) ≈ coef(fit_K, :mu) atol = 1e-4
end
