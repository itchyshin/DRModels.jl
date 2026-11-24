# Issue #1219: DRMControl(maxit, start, multi_start) on random-effect routes (drmTMB parity).
using DRModels, Test, Random, Statistics, Distributions

@testset "DRMControl multi_start on Gaussian (1 | g) location-scale" begin
    g = repeat(1:14, 6)
    n = length(g)
    rng = MersenneTwister(20261031)
    age = randn(rng, n)
    y = 1.0 .+ 0.3 .* age .+ 0.2 .* (age .^ 2) .+ 0.5 .* randn(rng, n) .+ randn(rng, 14)[g]
    d = (; y = Float64.(y), age = Float64.(age), Seed = string.(g))
    f = bf(@formula(y ~ age + age^2 + (1 | Seed)), @formula(sigma ~ age))
    base = drm(f, Gaussian(); data = d, control = DRMControl(se = false))
    ms = redirect_stderr(devnull) do
        drm(f, Gaussian(); data = d, control = DRMControl(multi_start = 5, maxit = 2000, se = false))
    end
    info = control_info(ms)
    @test length(info.start_logliks) == 5
    @test 1 <= info.kept_start <= 5
    @test loglikelihood(ms) >= loglikelihood(base) - 1e-6
    @test loglikelihood(ms) == maximum(filter(isfinite, info.start_logliks))
end

@testset "DRMControl maxit on Poisson (1 | g)" begin
    g = repeat(1:12, 8)
    n = length(g)
    rng = MersenneTwister(7)
    x = randn(rng, n)
    λ = exp.(0.2 .+ 0.25 .* x)
    y = Float64.([rand(rng, Distributions.Poisson(li)) for li in λ])
    d = (; y, x, g = string.(g))
    f = bf(@formula(y ~ 1 + x + (1 | g)))
    capped = redirect_stderr(devnull) do
        drm(f, DRModels.Poisson(); data = d, control = DRMControl(maxit = 3, se = false))
    end
    @test niterations(capped) <= 3
end